"""Controller v1: layered, physically consistent HVAC policy.

Layers, in the order they run each step:

1. Schedule   decide per AHU whether it is active (occupied hours, adaptive
              pre-start before the morning, unoccupied guard limits) or off.
2. Zone       per-zone PI on the distance outside a target band, giving a demand
              d in [-1, 1] (positive = heat, negative = cool).
3. Flow       demand -> requested airflow fraction per zone; the AHU static
              pressure is the lowest that lets the neediest zone reach its flow
              with its damper fully open (fan power ~ flow x pressure).
4. AHU        supply-air temperature reset from the worst cooling demand, a
              dry-bulb economizer, run command with minimum on-time.
5. Plant      hot-water pump runs only when some active zone needs heat; pump
              dp is the lowest that lets the neediest valve open fully; water
              temperature is a smooth function of outside air (no cycling).

Everything is derived from ``building_meta_data`` and the observation, so the
policy has no building- or weather-specific constants.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sb import params as P
from sb.obs import Layout, build_layout, decode, to_norm

SAT_MIN, SAT_MAX = 285.0, 305.0
SP_MAX = 2000.0
SP_FLOOR = 200.0
RECIRC_MAX = 0.7
TW_MIN, TW_MAX = 310.0, 350.0
DP_MAX = 20.0
DAMPER_FLOOR = 0.1


class ControllerV1:
    name = "ctrl_v1"

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any = None,
                 params: Mapping[str, float] | None = None) -> None:
        self.L: Layout = build_layout(building_meta_data)
        self.p = P.merged(dict(params) if params else None)
        V, K = self.L.V, self.L.K
        self.integ = np.zeros(V)
        self.prev_t: np.ndarray | None = None
        self.ahu_on = np.zeros(K, dtype=bool)
        self.ahu_on_steps = np.zeros(K, dtype=np.int64)
        self.occ_seen_today = False
        self.last_dow = -1
        self.comfort_lo, self.comfort_hi = self.L.comfort_window
        # zone membership matrix (K, V) for fast per-AHU max
        self.member = np.zeros((K, V), dtype=bool)
        self.member[self.L.zone_ahu, np.arange(V)] = True

    # ------------------------------------------------------------------ helpers
    def _hours_to_morning(self, hour: float, dow: int) -> float:
        """Hours until the next workday's comfort start (inf if > 24 h away)."""
        m = self.L.morning_hour
        if hour < m and dow < 5:
            return m - hour
        nxt = (dow + 1) % 7
        if hour >= m and nxt < 5:
            return 24.0 - hour + m
        return np.inf

    def _per_ahu_max(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.member, x[None, :], -np.inf).max(axis=1)

    # ------------------------------------------------------------------ policy
    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        p, L = self.p, self.L
        s = decode(obs, L)
        V, K = L.V, L.K

        # Lag-compensated zone temperature (sensor is one step stale).
        t = s.zone_t
        if self.prev_t is not None:
            t = t + p["lag_comp"] * (s.zone_t - self.prev_t)
        self.prev_t = s.zone_t.copy()

        # ---- 1. schedule
        if s.dow != self.last_dow:
            self.last_dow, self.occ_seen_today = s.dow, False
        if s.occupancy > 0.5:
            self.occ_seen_today = True
        occupied = s.comfort_now
        if occupied and s.hour >= p["holiday_check_h"] and not self.occ_seen_today:
            occupied = False  # weekday with nobody in by mid-morning: holiday
        lo_t = self.comfort_lo + p["margin_lo_K"]
        hi_t = self.comfort_hi - p["margin_hi_K"]

        if occupied:
            zone_active = np.ones(V, dtype=bool)
        else:
            # Adaptive pre-start per AHU: start when the time needed to bring its
            # worst zone into band reaches the time left before the morning.
            h_left = self._hours_to_morning(s.hour, s.dow)
            need_h = np.maximum((lo_t - t) / p["rate_heat_K_per_h"], (t - hi_t) / p["rate_cool_K_per_h"])
            need_h = np.maximum(need_h, 0.0)
            ahu_need = self._per_ahu_max(need_h) + p["prestart_buffer_h"]
            ahu_need = np.minimum(ahu_need, p["prestart_max_h"])
            prestart = h_left <= ahu_need
            # Unoccupied guard: keep zones from drifting beyond recoverable limits.
            guard = self._per_ahu_max(np.maximum(p["unocc_lo_K"] - t, t - p["unocc_hi_K"])) > 0
            zone_active = (prestart | guard)[L.zone_ahu]

        # In active-but-unoccupied guard mode, aim only at the guard limits.
        if occupied:
            z_lo, z_hi = np.full(V, lo_t), np.full(V, hi_t)
        else:
            h_left = self._hours_to_morning(s.hour, s.dow)
            pre = zone_active & (h_left <= p["prestart_max_h"] + p["prestart_buffer_h"])
            z_lo = np.where(pre, lo_t, p["unocc_lo_K"])
            z_hi = np.where(pre, hi_t, p["unocc_hi_K"])

        # ---- 2. zone PI demand (positive = heat)
        err = np.where(t < z_lo, z_lo - t, np.where(t > z_hi, z_hi - t, 0.0))
        err = np.where(zone_active, err, 0.0)
        self.integ += p["ki"] * err
        self.integ = np.where(err == 0.0, self.integ * (1.0 - p["i_leak"]), self.integ)
        self.integ = np.where(zone_active, np.clip(self.integ, -1.0, 1.0), 0.0)
        d = np.clip(p["kp"] * err + self.integ, -1.0, 1.0)
        heat = np.maximum(d, 0.0)
        cool = np.maximum(-d, 0.0)

        # ---- 4a. AHU run with minimum on-time
        dmax = self._per_ahu_max(np.abs(d) * zone_active)
        any_active = self._per_ahu_max(zone_active.astype(float)) > 0
        want_on = any_active & (dmax > p["ahu_on_thr"])
        keep_on = self.ahu_on & any_active & ((dmax > p["ahu_off_thr"]) | (self.ahu_on_steps < p["ahu_min_on_steps"]))
        on = want_on | keep_on
        self.ahu_on_steps = np.where(on, self.ahu_on_steps + 1, 0)
        self.ahu_on = on
        zone_on = on[L.zone_ahu]

        # ---- 3. flow allocation -> static pressure and dampers
        f = np.full(V, p["f_idle"])
        f = np.where(cool > 0, p["f_idle"] + cool * (p["f_cool_max"] - p["f_idle"]), f)
        f = np.where(heat > 0, p["f_heat_min"] + heat * (p["f_heat_max"] - p["f_heat_min"]), f)
        f = np.clip(f, 1e-3, 1.0)
        f_max = self._per_ahu_max(f)
        sp = np.clip(SP_MAX * f_max ** 2, SP_FLOOR, SP_MAX)
        scale = np.sqrt(sp / SP_MAX)[L.zone_ahu]
        damper = np.clip(f / scale, DAMPER_FLOOR, 1.0)

        # ---- 4b. supply-air temperature and economizer
        cool_max = self._per_ahu_max(cool)
        sat = p["sat_idle_K"] - cool_max * (p["sat_idle_K"] - p["sat_cool_K"])
        sat = np.clip(sat, SAT_MIN, SAT_MAX)
        t_ret = np.array([t[self.member[k]].mean() for k in range(K)])
        oat = s.oat
        recirc = np.full(K, RECIRC_MAX)
        econ = (oat < t_ret - p["econ_margin_K"]) & (cool_max > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_mix = (sat - oat) / (t_ret - oat)
        recirc = np.where(econ, np.clip(r_mix, 0.0, RECIRC_MAX), recirc)

        # ---- 5. hot water: valves, pump dp, water temperature
        heat_eff = np.where(zone_on, heat, 0.0)
        h_max = float(heat_eff.max()) if V else 0.0
        pump_on = h_max > 1e-3
        if pump_on:
            dp = float(np.clip(DP_MAX * h_max ** 2, p["dp_min_frac"] * DP_MAX, DP_MAX))
            valve = np.clip(heat_eff / np.sqrt(dp / DP_MAX), 0.0, 1.0)
        else:
            dp = 0.0
            valve = np.zeros(V)
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


# Name the Kaggle template expects to find.
MyPolicy = ControllerV1
