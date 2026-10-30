"""Decode the flat observation vector into physical units.

Built once per building from ``building_meta_data``; ``decode(obs)`` is then a
handful of vectorized numpy gathers, cheap enough to call every step.

Scaling rules (verified empirically on the shipped scenarios, see
``scripts/check_obs.py``):

* zone air temperature        normalized: T = 295 + 5 x            [K]
* VAV damper / reheat         normalized: u = 0.5 + 0.5 x          [0..1]
* zone heating/cooling setpoint sensors      raw K
* every ``ahs_*`` (air handler) field        raw (K, Pa, fraction, counts)
* hws supply water temperature               raw K
* hws differential pressure setpoint         normalized: 10 + 6 x
* time: 12 phase-shifted sin/cos pairs; phase 000 is the true angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class Layout:
    """Index tables for one building. Zones are in VAV order of ``vav_ids``."""

    vav_ids: list[str]
    ahu_ids: list[str]
    zone_ahu: np.ndarray            # (V,) AHU index serving each zone
    # observation indices
    i_zone_t: np.ndarray            # (V,)
    i_zone_hsp: np.ndarray          # (V,) heating setpoint sensor (raw K)
    i_zone_csp: np.ndarray          # (V,) cooling setpoint sensor (raw K)
    i_zone_damper: np.ndarray       # (V,)
    i_zone_reheat: np.ndarray       # (V,)
    i_ahu_oat: np.ndarray           # (K,)
    i_ahu_sat: np.ndarray           # (K,) supply air temp sensor
    i_ahu_flow: np.ndarray          # (K,) supply air flow sensor
    i_ahu_run: np.ndarray           # (K,) supervisor run command echo
    i_hws_t: int
    i_hod_sin: int
    i_hod_cos: int
    i_dow_sin: int
    i_dow_cos: int
    i_occ: int
    i_comfort_now: int
    i_comfort_soon: int
    # action indices
    a_ahu_temp: np.ndarray          # (K,)
    a_ahu_pressure: np.ndarray      # (K,)
    a_ahu_recirc: np.ndarray        # (K,)
    a_pump_dp: int
    a_water_temp: int
    a_damper: np.ndarray            # (V,)
    a_reheat: np.ndarray            # (V,)
    n_cont: int
    n_disc: int
    # schedule
    morning_hour: float
    evening_hour: float
    comfort_window: tuple[float, float]
    eco_window: tuple[float, float]

    @property
    def V(self) -> int:
        return len(self.vav_ids)

    @property
    def K(self) -> int:
        return len(self.ahu_ids)


def _num(s: str) -> int:
    return int(s.rsplit("_", 1)[-1])


def build_layout(meta: Mapping[str, Any]) -> Layout:
    names = list(meta["observation_field_names"])
    pos = {n: i for i, n in enumerate(names)}

    def idx(name: str) -> int:
        if name not in pos:
            raise KeyError(f"observation field not found: {name}")
        return pos[name]

    zai = meta["zone_action_indices"]
    vav_ids = sorted(zai.keys(), key=_num)
    assignments = meta["ahu_zone_assignments"]
    # AHU order of the positional action lists (temp/pressure/recirc and the
    # discrete run commands). Taken from the action normalizer key order, which
    # follows the environment's device order; numeric sort is the fallback.
    suffix = "_supply_air_temperature_setpoint"
    ahu_ids = [k[: -len(suffix)] for k in meta.get("action_normalization_metadata", {})
               if k.startswith("ahu_") and k.endswith(suffix)]
    if sorted(ahu_ids) != sorted(assignments.keys()):
        ahu_ids = sorted(assignments.keys(), key=_num)
    room_to_ahu = {}
    for k, ahu in enumerate(ahu_ids):
        for room in assignments[ahu]:
            room_to_ahu[_num(room)] = k
    zone_ahu = np.array([room_to_ahu.get(_num(v), 0) for v in vav_ids], dtype=np.int64)

    def zone_field(suffix):
        return np.array([idx(f"{v}_{suffix}") for v in vav_ids], dtype=np.int64)

    def ahu_field(suffix):
        return np.array([idx(f"ahs_{a}_{suffix}") for a in ahu_ids], dtype=np.int64)

    temp_idx = list(meta["temp_indices"])
    pres_idx = list(meta["pressure_indices"])
    K = len(ahu_ids)
    assert len(temp_idx) == K and len(pres_idx) == K + 1, (len(temp_idx), len(pres_idx), K)

    return Layout(
        vav_ids=vav_ids,
        ahu_ids=ahu_ids,
        zone_ahu=zone_ahu,
        i_zone_t=zone_field("zone_air_temperature_sensor"),
        i_zone_hsp=zone_field("zone_air_heating_temperature_setpoint_sensor"),
        i_zone_csp=zone_field("zone_air_cooling_temperature_setpoint_sensor"),
        i_zone_damper=zone_field("supply_air_damper_percentage_command"),
        i_zone_reheat=zone_field("reheat_valve_percentage_command"),
        i_ahu_oat=ahu_field("outside_air_temperature_sensor"),
        i_ahu_sat=ahu_field("supply_air_temperature_sensor"),
        i_ahu_flow=ahu_field("supply_air_flowrate_sensor"),
        i_ahu_run=ahu_field("supervisor_run_command"),
        i_hws_t=idx("hws_supply_water_temperature_sensor"),
        i_hod_sin=idx("hod_sin_000"),
        i_hod_cos=idx("hod_cos_000"),
        i_dow_sin=idx("dow_sin_000"),
        i_dow_cos=idx("dow_cos_000"),
        i_occ=idx("num_occupants"),
        i_comfort_now=idx("comfort_mode_now"),
        i_comfort_soon=idx("comfort_mode_soon"),
        a_ahu_temp=np.array(temp_idx, dtype=np.int64),
        a_ahu_pressure=np.array(pres_idx[:K], dtype=np.int64),
        a_ahu_recirc=np.array(meta["recirculation_indices"], dtype=np.int64),
        a_pump_dp=int(pres_idx[K]),
        a_water_temp=int(meta["water_indices"][0]),
        a_damper=np.array([zai[v]["supply_air_damper_percentage_command"] for v in vav_ids], dtype=np.int64),
        a_reheat=np.array([zai[v]["reheat_valve_percentage_command"] for v in vav_ids], dtype=np.int64),
        n_cont=int(meta["n_continuous_action"]),
        n_disc=int(meta["n_discrete_action"]),
        morning_hour=float(meta["morning_start_hour"]),
        evening_hour=float(meta["evening_start_hour"]),
        comfort_window=tuple(meta["comfort_temp_window"]),
        eco_window=tuple(meta["eco_temp_window"]),
    )


@dataclass
class State:
    """One decoded observation, physical units."""

    zone_t: np.ndarray      # (V,) K
    zone_lo: np.ndarray     # (V,) K, heating setpoint (lower comfort edge)
    zone_hi: np.ndarray     # (V,) K, cooling setpoint (upper comfort edge)
    damper: np.ndarray      # (V,) 0..1 as realized
    reheat: np.ndarray      # (V,) 0..1 as realized
    oat: float              # K, outside air
    sat: np.ndarray         # (K,) K, supply air temp
    hws_t: float            # K
    hour: float             # 0..24 local wall clock
    dow: int                # 0=Mon .. 6=Sun
    occupancy: float        # raw building occupancy feature
    comfort_now: bool
    comfort_soon: bool


def decode(obs: np.ndarray, L: Layout) -> State:
    o = np.asarray(obs, dtype=np.float64)
    hour = (np.arctan2(o[L.i_hod_sin], o[L.i_hod_cos]) % TWO_PI) * 24.0 / TWO_PI
    dow = int(round((np.arctan2(o[L.i_dow_sin], o[L.i_dow_cos]) % TWO_PI) * 7.0 / TWO_PI)) % 7
    return State(
        zone_t=295.0 + 5.0 * o[L.i_zone_t],
        zone_lo=o[L.i_zone_hsp],
        zone_hi=o[L.i_zone_csp],
        damper=0.5 + 0.5 * o[L.i_zone_damper],
        reheat=0.5 + 0.5 * o[L.i_zone_reheat],
        oat=float(np.median(o[L.i_ahu_oat])),
        sat=o[L.i_ahu_sat],
        hws_t=float(o[L.i_hws_t]),
        hour=float(hour),
        dow=dow,
        occupancy=float(o[L.i_occ]),
        comfort_now=bool(o[L.i_comfort_now] > 0.5),
        comfort_soon=bool(o[L.i_comfort_soon] > 0.5),
    )


def to_norm(native, lo: float, hi: float):
    """Map a native value in [lo, hi] to the [-1, 1] action range (clipped)."""
    return np.clip(2.0 * (np.asarray(native, dtype=np.float64) - lo) / (hi - lo) - 1.0, -1.0, 1.0)
