"""Run the built submission notebook against a stand-in for the Kaggle server.

    python scripts/test_notebook.py submission/notebook.ipynb building_train_8/denver_6 2016

The real gateway cannot be downloaded before the competition rules are
accepted, so this stub reproduces its contract as the template shows it: a
fresh Policy(building_meta_data, action_spec) per scenario, get_action(obs)
every step, and any action outside the spec counts as an invalid submission.
Metadata goes through a list/tuple round trip to mimic serialization. The
script checks that the embedded copy of our package is the one imported.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

KIT = Path("/data/smartbuild/starter-kit")

STUB = '''
import abc, os, time
import numpy as np


class Policy(abc.ABC):
    def __init__(self, building_meta_data, action_spec):
        self.building_meta_data = building_meta_data
        self.action_spec = action_spec

    @abc.abstractmethod
    def get_action(self, obs): ...


def _roundtrip(x):
    if isinstance(x, dict):
        return {k: _roundtrip(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_roundtrip(v) for v in x]
    return x


class SmartBuildingsInferenceServer:
    def __init__(self, policy_cls):
        self.policy_cls = policy_cls

    def run(self, competition_data_folder=None):
        from pathlib import Path
        from smart_buildings import evaluate as ev
        scen = Path(os.environ["SB_TEST_SCENARIO"])
        steps = int(os.environ["SB_TEST_STEPS"])
        env = ev.load_hybrid_environment(scen / "config.gin")
        meta = _roundtrip(ev.create_building_meta_data(env, scen))
        spec = env.action_spec()
        policy = self.policy_cls(meta, spec)
        obs, _ = env.reset()
        total, t_pol, n = 0.0, 0.0, 0
        c_sp, d_sp = spec["continuous_action"], spec["discrete_action"]
        for _ in range(steps):
            t0 = time.perf_counter()
            a = policy.get_action(obs)
            t_pol += time.perf_counter() - t0
            c, d = a["continuous_action"], a["discrete_action"]
            assert c.dtype == np.float32 and d.dtype == np.int32, (c.dtype, d.dtype)
            assert c.shape == c_sp.shape and d.shape == d_sp.shape, (c.shape, d.shape)
            assert np.all(c >= c_sp.low) and np.all(c <= c_sp.high), "continuous action out of range"
            assert np.all(d >= d_sp.low) and np.all(d <= d_sp.high), "discrete action out of range"
            obs, r, term, trunc, _ = env.step(a)
            assert np.isfinite(r), r
            total += float(r); n += 1
            if term or trunc:
                break
        self.result = {"steps": n, "mean_reward": total / n, "policy_ms_per_step": 1000 * t_pol / n}
        print("STUB RESULT", self.result, flush=True)
'''


def main():
    nb_path, scen, steps = sys.argv[1], sys.argv[2], sys.argv[3]
    nb = json.loads(Path(nb_path).read_text())
    stub_root = Path(tempfile.mkdtemp(prefix="kaggle_stub_"))
    (stub_root / "kaggle_evaluation").mkdir()
    (stub_root / "kaggle_evaluation" / "__init__.py").write_text("")
    (stub_root / "kaggle_evaluation" / "smart_buildings_inference_server.py").write_text(STUB)
    os.environ["SB_TEST_SCENARIO"] = str(KIT / "smart_buildings/training_scenarios" / scen)
    os.environ["SB_TEST_STEPS"] = steps
    os.chdir(KIT)
    sys.path.insert(0, str(stub_root))
    g = {"__name__": "__main__"}
    t0 = time.time()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            exec(compile("".join(c["source"]), "<notebook>", "exec"), g)
    import sb
    print("sb imported from", sb.__file__)
    assert sb.__file__.startswith("/tmp/sb_pkg"), "notebook used the installed package, not its embedded copy"
    print(f"notebook ran in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
