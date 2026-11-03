"""Controller v3: v2 with per-zone cooling capacity that adapts to each zone.

v2 maps full PI demand to one fixed heating and one fixed cooling rate for every
zone (q_heat_max_W, q_cool_max_W). Zones differ in floor area and exposure, so
a cap tuned on the training buildings under-serves big rooms in hot weather:
raising the cooling cap from 9.7 to 20 kW took one large holdout building in a
hot climate from 0.0213 to 0.0183 and removed nearly all of its comfort loss.

Each zone carries its own cooling and heating scale, starting at 1. Scaling a
zone's watts also scales its loop gain, which matches a bigger zone's bigger
thermal capacity.

* Cooling: while the zone is above its target band with cooling demand at or
  above cap_level and its AHU running, the scale grows by cap_rate per step up
  to cap_max.
* Heating: grows only on true saturation (demand pinned at full) and only up to
  cap_max_heat, which defaults to 1 (no growth). Growing heating at a 0.6
  trigger made the Fairbanks week far worse (-91.1 to -153.3): the extra gain
  overshoots in the cold.
* Otherwise each scale relaxes toward 1 by cap_relax per step.

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
        up, relax = 1.0 + p["cap_rate"], 1.0 - p["cap_relax"]
        grow_c = served & (cool >= min(p["cap_level"], SATURATED)) & (err < 0)
        grow_h = served & (heat >= SATURATED) & (err > 0)
        self.k_cool = np.where(grow_c, np.minimum(self.k_cool * up, p["cap_max"]), 1.0 + (self.k_cool - 1.0) * relax)
        self.k_heat = np.where(grow_h, np.minimum(self.k_heat * up, p["cap_max_heat"]),
                               1.0 + (self.k_heat - 1.0) * relax)
        return heat * p["q_heat_max_W"] * self.k_heat, cool * p["q_cool_max_W"] * self.k_cool
