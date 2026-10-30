"""Controller parameters: defaults and search bounds (single source of truth).

Every tunable constant of the policy lives here so CMA-ES can optimize them as
one vector. Units are in the names: K, h (hours), frac (0..1), per_step.
"""

from __future__ import annotations

# name: (default, low, high)
SPEC: dict[str, tuple[float, float, float]] = {
    # --- comfort targets during active (occupied / pre-start) mode
    "margin_lo_K": (0.6, 0.0, 1.5),        # target lower edge = comfort_lo + margin
    "margin_hi_K": (0.6, 0.0, 1.5),        # target upper edge = comfort_hi - margin
    "lag_comp": (0.0, 0.0, 1.5),           # extrapolate zone temp by lag_comp * last delta
    # --- per-zone PI demand (positive = heat, negative = cool)
    # Zone air settles in ~2.5 min at full flow, faster than the 5-min step, and
    # the sensor lags a step, so high gains oscillate between full heat and cool.
    "kp": (0.07, 0.01, 0.5),               # demand per K of error
    "ki": (0.0035, 0.0002, 0.05),          # demand per K per step
    "i_leak": (0.002, 0.0, 0.05),          # integrator leak per step while in band
    # --- flow allocation (fraction of 2 m3/s design flow at 2000 Pa)
    "f_idle": (0.08, 0.03, 0.4),           # zone in band
    "f_heat_min": (0.25, 0.05, 0.8),       # heating zone at zero demand
    "f_heat_max": (0.6, 0.1, 1.0),         # heating zone at full demand
    "f_cool_max": (1.0, 0.3, 1.0),         # cooling zone at full demand
    # --- AHU
    "sat_cool_K": (286.0, 285.0, 295.0),   # supply air setpoint at full cooling demand
    "sat_idle_K": (293.0, 288.0, 305.0),   # supply air setpoint with no cooling demand
    "ahu_on_thr": (0.05, 0.0, 0.5),        # max |demand| that switches an AHU on
    "ahu_off_thr": (0.02, 0.0, 0.5),       # max |demand| below which it may switch off
    "ahu_min_on_steps": (6.0, 1.0, 36.0),
    "econ_margin_K": (1.0, 0.0, 5.0),      # outside air must be this much colder than return to economize
    # --- hot water
    "tw_cold_K": (345.0, 320.0, 350.0),    # water temp setpoint when OAT <= oat_cold
    "tw_warm_K": (330.0, 310.0, 350.0),    # water temp setpoint when OAT >= oat_warm
    "oat_cold_K": (263.0, 230.0, 285.0),
    "oat_warm_K": (288.0, 270.0, 300.0),
    "dp_min_frac": (0.05, 0.0, 0.5),       # minimum pump dp (fraction of 20 bar) while pump runs
    # --- schedule / pre-start
    "rate_heat_K_per_h": (2.0, 0.3, 8.0),  # assumed recovery rate when heating
    "rate_cool_K_per_h": (2.0, 0.3, 8.0),  # assumed recovery rate when cooling
    "prestart_buffer_h": (0.5, 0.0, 3.0),
    "prestart_max_h": (4.0, 1.0, 8.0),
    "unocc_lo_K": (287.0, 278.0, 293.0),   # unoccupied: heat if any zone below this
    "unocc_hi_K": (303.0, 298.0, 312.0),   # unoccupied: cool if any zone above this
    "holiday_check_h": (10.0, 8.0, 13.0),  # weekday hour by which zero occupancy means holiday
}

DEFAULTS: dict[str, float] = {k: v[0] for k, v in SPEC.items()}
NAMES: list[str] = list(SPEC)


def merged(overrides: dict | None) -> dict[str, float]:
    p = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(SPEC)
        if unknown:
            raise KeyError(f"unknown params: {sorted(unknown)}")
        p.update({k: float(v) for k, v in overrides.items()})
    return p


def clip(p: dict[str, float]) -> dict[str, float]:
    return {k: min(max(v, SPEC[k][1]), SPEC[k][2]) for k, v in p.items()}
