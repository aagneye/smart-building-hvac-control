"""Controller v2: v1's schedule and zone PI, with model-based air and water sizing.

v1 mapped heating demand straight to a large airflow, so in cold climates most
of the heat went into the 30% outside air the AHU must take in. v2 treats the
PI output as a heat (or cooling) rate in watts and sizes the actuators from the
plant equations:

* heating air: just enough flow to deliver the heat at a discharge temperature
  near the 308.15 K cap, since every m3/s carries outside air that must be
  reheated;
* cooling air: the flow that removes the heat at the AHU's actual supply
  temperature (the mixed air when that is already colder than the setpoint);
* hot water: the coil water flow that lifts that air to the needed discharge
  temperature, then the lowest pump dp that lets the neediest valve open fully.

The integrator is frozen, not reset, while a zone is inactive, so each morning
starts from the previous day's steady-state demand.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sb.obs import decode, to_norm
from sb.policy import (DAMPER_FLOOR, DP_MAX, RECIRC_MAX, SAT_MAX, SAT_MIN, SP_FLOOR, SP_MAX, TW_MAX, TW_MIN,
                       ControllerV1)

RHO_CP_AIR = 1.2 * 1006.0        # J / (m3 K)
RHO_CP_WATER = 1000.0 * 4180.0   # J / (m3 K)
REHEAT_KV = 0.108                # m3/h per sqrt(dp) at a fully open valve (shipped configs)
T_DIS_MAX = 308.15               # VAV discharge temperature cap, K
DEFAULT_VAV_FLOW = 2.0           # m3/s at 2000 Pa with the damper fully open


class ControllerV2(ControllerV1):
    name = "ctrl_v2"

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any = None,
                 params: Mapping[str, float] | None = None) -> None:
        super().__init__(building_meta_data, action_spec, params)
        names = list(building_meta_data["observation_field_names"])
        pos = {n: i for i, n in enumerate(names)}
        idx = [pos.get(f"{v}_supply_air_flowrate_setpoint") for v in self.L.vav_ids]
        self.i_vmax = np.array(idx, dtype=np.int64) if all(i is not None for i in idx) else None
        self.vmax: np.ndarray | None = None

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        p, L = self.p, self.L
        o = np.asarray(obs, dtype=np.float64)
        s = decode(o, L)
        V, K = L.V, L.K
        if self.vmax is None:
            v = o[self.i_vmax] if self.i_vmax is not None else np.full(V, DEFAULT_VAV_FLOW)
            self.vmax = np.where(np.isfinite(v) & (v > 0.05), v, DEFAULT_VAV_FLOW)

        t = s.zone_t
        if self.prev_t is not None:
            t = t + p["lag_comp"] * (s.zone_t - self.prev_t)
        self.prev_t = s.zone_t.copy()

        # ---- schedule (as v1)
        if s.dow != self.last_dow:
            self.last_dow, self.occ_seen_today = s.dow, False
        if s.occupancy > 0.5:
            self.occ_seen_today = True
        occupied = s.comfort_now
        if occupied and s.hour >= p["holiday_check_h"] and not self.occ_seen_today:
            occupied = False
        lo_t = self.comfort_lo + p["margin_lo_K"]
        hi_t = self.comfort_hi - p["margin_hi_K"]
        if occupied:
            zone_active = np.ones(V, dtype=bool)
            z_lo, z_hi = np.full(V, lo_t), np.full(V, hi_t)
        else:
            h_left = self._hours_to_morning(s.hour, s.dow)
            need_h = np.maximum((lo_t - t) / p["rate_heat_K_per_h"], (t - hi_t) / p["rate_cool_K_per_h"])
            ahu_need = np.minimum(self._per_ahu_max(np.maximum(need_h, 0.0)) + p["prestart_buffer_h"],
                                  p["prestart_max_h"])
            prestart = h_left <= ahu_need
            guard = self._per_ahu_max(np.maximum(p["unocc_lo_K"] - t, t - p["unocc_hi_K"])) > 0
            zone_active = (prestart | guard)[L.zone_ahu]
            pre = zone_active & (h_left <= p["prestart_max_h"] + p["prestart_buffer_h"])
            z_lo = np.where(pre, lo_t, p["unocc_lo_K"])
            z_hi = np.where(pre, hi_t, p["unocc_hi_K"])

        # ---- zone PI demand; integrator frozen while inactive
        err = np.where(t < z_lo, z_lo - t, np.where(t > z_hi, z_hi - t, 0.0))
        err = np.where(zone_active, err, 0.0)
        integ = self.integ + p["ki"] * err
        integ = np.where(zone_active & (err == 0.0), integ * (1.0 - p["i_leak"]), integ)
        self.integ = np.clip(integ, -1.0, 1.0)
        d = np.where(zone_active, np.clip(p["kp"] * err + self.integ, -1.0, 1.0), 0.0)
        heat = np.maximum(d, 0.0)
        cool = np.maximum(-d, 0.0)

        # ---- AHU run with minimum on-time (as v1)
        dmax = self._per_ahu_max(np.abs(d))
        any_active = self._per_ahu_max(zone_active.astype(float)) > 0
        want_on = any_active & (dmax > p["ahu_on_thr"])
        keep_on = self.ahu_on & any_active & ((dmax > p["ahu_off_thr"]) | (self.ahu_on_steps < p["ahu_min_on_steps"]))
        on = want_on | keep_on
        self.ahu_on_steps = np.where(on, self.ahu_on_steps + 1, 0)
        self.ahu_on = on
        zone_on = on[L.zone_ahu]

        # ---- supply air temperature, economizer, actual supply temperature
        cool_max = self._per_ahu_max(cool)
        sat = np.clip(p["sat_idle_K"] - cool_max * (p["sat_idle_K"] - p["sat_cool_K"]), SAT_MIN, SAT_MAX)
        t_ret = np.array([t[self.member[k]].mean() for k in range(K)])
        oat = s.oat
        econ = (oat < t_ret - p["econ_margin_K"]) & (cool_max > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_mix = (sat - oat) / (t_ret - oat)
        recirc = np.where(econ, np.clip(np.nan_to_num(r_mix, nan=RECIRC_MAX), 0.0, RECIRC_MAX), RECIRC_MAX)
        t_mix = recirc * t_ret + (1.0 - recirc) * oat
        t_sa_z = np.minimum(t_mix, sat)[L.zone_ahu]

        # ---- air sizing from the heat or cooling rate
        q_h = heat * p["q_heat_max_W"]
        q_c = cool * p["q_cool_max_W"]
        t_dis = T_DIS_MAX - p["dis_margin_K"]
        c_air = np.where(heat > 0, q_h / np.maximum(t_dis - t, 1.0), q_c / np.maximum(t - t_sa_z, 1.0))
        f = np.clip(c_air / RHO_CP_AIR / self.vmax, p["f_floor"], 1.0)
        f_max = self._per_ahu_max(f)
        sp = np.clip(SP_MAX * f_max ** 2, SP_FLOOR, SP_MAX)
        scale = np.sqrt(sp / SP_MAX)[L.zone_ahu]
        damper = np.clip(f / scale, DAMPER_FLOOR, 1.0)
        flow = np.maximum(damper * scale * self.vmax, 1e-4)

        # ---- water sizing: coil flow that lifts the supply air to the needed discharge
        heat_on = (heat > 0) & zone_on
        t_need = np.minimum(t + q_h / (RHO_CP_AIR * flow), T_DIS_MAX)
        coil_w = RHO_CP_AIR * flow * np.maximum(t_need - t_sa_z, 0.0)
        c_w = np.where(heat_on, coil_w / np.maximum(s.hws_t - t_sa_z, 5.0) * p["water_margin"], 0.0)
        g = c_w / RHO_CP_WATER * 3600.0 / REHEAT_KV          # valve * sqrt(dp)
        g_max = float(g.max()) if V else 0.0
        pump_on = g_max > 1e-6
        if pump_on:
            dp = float(np.clip(g_max ** 2, p["dp_min_frac"] * DP_MAX, DP_MAX))
            valve = np.clip(g / np.sqrt(dp), 0.0, 1.0)
        else:
            dp, valve = 0.0, np.zeros(V)
        w = np.clip((oat - p["oat_cold_K"]) / max(p["oat_warm_K"] - p["oat_cold_K"], 1e-3), 0.0, 1.0)
        tw = p["tw_cold_K"] + w * (p["tw_warm_K"] - p["tw_cold_K"])

        # ---- encode
        cont = np.zeros(L.n_cont, dtype=np.float64)
        cont[L.a_ahu_temp] = to_norm(sat, SAT_MIN, SAT_MAX)
        cont[L.a_ahu_pressure] = to_norm(sp, 0.0, SP_MAX)
        cont[L.a_ahu_recirc] = to_norm(recirc, 0.0, 1.0)
        cont[L.a_pump_dp] = to_norm(dp, 0.0, DP_MAX)
        cont[L.a_water_temp] = to_norm(tw, TW_MIN, TW_MAX)
        cont[L.a_damper] = to_norm(np.where(zone_on, damper, DAMPER_FLOOR), 0.0, 1.0)
        cont[L.a_reheat] = to_norm(valve, 0.0, 1.0)
        disc = np.zeros(L.n_disc, dtype=np.int64)
        disc[:K] = on.astype(np.int64)
        disc[K] = int(pump_on)
        return {"continuous_action": cont.astype(np.float32), "discrete_action": disc.astype(np.int32)}
