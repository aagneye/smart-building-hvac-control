"""Build the Kaggle submission notebook from our package.

    python scripts/build_notebook.py --policy sb.policy_v2:ControllerV2 \
        --params params/v2_tuned_g22.json --slug sb-hvac-controller-v2 --out submission

Embeds the sb modules as source strings, so the notebook runs with internet
disabled. The policy wrapper follows the official template: it subclasses
``smart_buildings_inference_server.Policy`` and starts
``SmartBuildingsInferenceServer(...).run(competition_data_folder=...)``.
Actions are clipped into the action spec, and any exception inside the
controller falls back to the last valid action, so a bug cannot invalidate a
submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["sb/__init__.py", "sb/obs.py", "sb/params.py", "sb/policy.py", "sb/policy_v2.py"]
COMP = "neurips-2026-smart-buildings"
DATA = f"/kaggle/input/competitions/{COMP}"
# Same image as the official template notebook.
DOCKER = "gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a"

POLICY_CELL = '''\
import sys
sys.path.append("{data}/kaggle_evaluation")
sys.path.append("{data}")

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server
from {module} import {cls} as Controller

COMPETITION_DATA_FOLDER = "{data}/"
PARAMS = {params}


class MyPolicy(smart_buildings_inference_server.Policy):
    """Wraps our controller: clips into the action spec, never raises."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        cont, disc = action_spec["continuous_action"], action_spec["discrete_action"]
        self._c_lo = np.asarray(cont.low, dtype=np.float32)
        self._c_hi = np.asarray(cont.high, dtype=np.float32)
        self._d_lo, self._d_hi = np.asarray(disc.low), np.asarray(disc.high)
        self._last = {{
            "continuous_action": np.clip(np.zeros(cont.shape, dtype=np.float32), self._c_lo, self._c_hi),
            "discrete_action": np.clip(np.ones(disc.shape), self._d_lo, self._d_hi).astype(np.int32),
        }}
        self._errors = 0
        try:
            self._inner = Controller(building_meta_data, action_spec, PARAMS)
            print(f"Controller ready: {{len(building_meta_data['zone_temp_indices'])}} zones", flush=True)
        except Exception as exc:  # keep the submission valid even if setup fails
            self._inner = None
            print("controller setup failed, using fallback action:", repr(exc), flush=True)

    def get_action(self, obs):
        if self._inner is not None:
            try:
                a = self._inner.get_action(np.asarray(obs))
                c = np.nan_to_num(np.asarray(a["continuous_action"], dtype=np.float32))
                d = np.asarray(a["discrete_action"])
                self._last = {{
                    "continuous_action": np.clip(c, self._c_lo, self._c_hi).astype(np.float32),
                    "discrete_action": np.clip(d, self._d_lo, self._d_hi).astype(np.int32),
                }}
            except Exception as exc:
                self._errors += 1
                if self._errors <= 3:
                    print("controller error, reusing last action:", repr(exc), flush=True)
        return self._last


smart_buildings_inference_server.SmartBuildingsInferenceServer(MyPolicy).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
'''

PKG_CELL = '''\
# Our controller package (github.com/aagneye/smart-building-hvac-control),
# embedded so the notebook runs with internet disabled.
import os
import sys

_FILES = {files}
_ROOT = "/tmp/sb_pkg"
for _rel, _src in _FILES.items():
    _path = os.path.join(_ROOT, _rel)
    os.makedirs(os.path.dirname(_path), exist_ok=True)
    with open(_path, "w") as _f:
        _f.write(_src)
sys.path.insert(0, _ROOT)
'''


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def build(policy: str, params_path: str | None, slug: str, title: str, out: Path, username: str) -> Path:
    module, cls = policy.split(":")
    files = {m: (ROOT / "src" / m).read_text() for m in MODULES}
    params = json.loads(Path(params_path).read_text()) if params_path else {}
    params.pop("_score", None)
    nb = {
        "cells": [
            cell("markdown", f"# {title}\n\nPolicy `{policy}`, params `{params_path or 'defaults'}`.\n"),
            cell("code", PKG_CELL.format(files=json.dumps(files, indent=0))),
            cell("code", POLICY_CELL.format(data=DATA, module=module, cls=cls, params=json.dumps(params, indent=1))),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [COMP],
        "model_sources": [],
        "docker_image": DOCKER,
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out / "notebook.ipynb"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="sb.policy_v2:ControllerV2")
    ap.add_argument("--params")
    ap.add_argument("--slug", default="sb-hvac-controller-v2")
    ap.add_argument("--title", help="defaults to the slug, which keeps Kaggle's id and title in step")
    ap.add_argument("--username", default="aagneye")
    ap.add_argument("--out", default="submission")
    a = ap.parse_args()
    path = build(a.policy, a.params, a.slug, a.title or a.slug, Path(a.out), a.username)
    print("wrote", path, "and", path.parent / "kernel-metadata.json")


if __name__ == "__main__":
    main()
