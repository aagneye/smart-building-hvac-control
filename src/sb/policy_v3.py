"""Controller v3: v2 with per-zone capacity that adapts to each zone.

v2 maps full PI demand to one fixed heating and one fixed cooling rate for every
zone (q_heat_max_W, q_cool_max_W). Zones differ in floor area and exposure, so
a cap tuned on the training buildings under-serves big rooms in hot weather:
raising the cooling cap from 9.7 to 20 kW took one large holdout building in a
hot climate from 0.0213 to 0.0183 and removed nearly all of its comfort loss.

Here each zone carries its own heating and cooling scale, starting at 1. While
the zone sits outside its target band with the PI output at or above cap_level
(default: pinned at full demand) in that direction, with its AHU running, the
scale grows by cap_rate per step up to cap_max; otherwise it relaxes toward 1
by cap_relax per step. Scaling a
zone's watts also scales its loop gain, which matches a bigger zone's bigger
thermal capacity, so the closed loop keeps roughly the same dynamics.

With cap_rate = 0 this is exactly v2.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sb.policy_v2 import ControllerV2

SATURATED = 0.98  # PI output treated as pinned at full demand


class ControllerV3(ControllerV2):
    name = "ctrl_v3"

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any = None,
                 params: Mapping[str, float] | None = None) -> None:
        super().__init__(building_meta_data, action_spec, params)
        self.k_heat = np.ones(self.L.V)
        self.k_cool = np.ones(self.L.V)

    def _zone_rates(self, heat: np.ndarray, cool: np.ndarray, err: np.ndarray,
                    served: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        up, relax, k_max = 1.0 + p["cap_rate"], 1.0 - p["cap_relax"], p["cap_max"]
        level = min(p["cap_level"], SATURATED)
        grow_h = served & (heat >= level) & (err > 0)
        grow_c = served & (cool >= level) & (err < 0)
        self.k_heat = np.where(grow_h, np.minimum(self.k_heat * up, k_max), 1.0 + (self.k_heat - 1.0) * relax)
        self.k_cool = np.where(grow_c, np.minimum(self.k_cool * up, k_max), 1.0 + (self.k_cool - 1.0) * relax)
        return heat * p["q_heat_max_W"] * self.k_heat, cool * p["q_cool_max_W"] * self.k_cool
