"""Run the built notebook against the real Kaggle gateway on the VM.

    /data/smartbuild/.venv-gw/bin/python scripts/test_notebook_gateway.py \
        submission/notebook.ipynb /data/smartbuild/kaggle/comp/data [max_steps]

Rewrites the notebook's /kaggle/input path to the local copy of the
competition data, optionally truncates each scenario with the gateway's own
HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS, runs every code cell in a scratch
directory exactly as Kaggle's commit run would, and prints the
submission.parquet the gateway writes.

Use the separate .venv-gw environment: the gateway needs grpcio-tools and
polars, and imports the simulator from the competition data folder.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

KAGGLE_DATA = "/kaggle/input/competitions/neurips-2026-smart-buildings"


def main() -> None:
    nb_path = Path(sys.argv[1]).resolve()
    data = Path(sys.argv[2]).resolve()
    if len(sys.argv) > 3:
        os.environ["HEARTH_HOST_SMART_BUILDINGS_MAX_STEPS"] = sys.argv[3]
    nb = json.loads(nb_path.read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = src.replace(KAGGLE_DATA, str(data))
    work = Path(tempfile.mkdtemp(prefix="nb_gateway_"))
    os.chdir(work)
    t0 = time.time()
    exec(compile(src, "<notebook>", "exec"), {"__name__": "__main__"})
    print(f"notebook finished in {time.time() - t0:.0f}s in {work}")
    import polars as pl

    print(pl.read_parquet(work / "submission.parquet"))


if __name__ == "__main__":
    main()
