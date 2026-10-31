"""Build the wider tuning and holdout set for robustness to unseen buildings and climates.

Run on the VM from the workspace root:

    .venv/bin/python scripts/make_wide_set.py

Everything goes into the starter kit (paths in config.gin are relative to it):

* Synthetic climates: real station series with the temperature shifted and the
  day-night swing stretched around a 24 h rolling mean, written as
  weather/csv/syn_<name>.csv plus presets syn_<name>_1..8 that reuse the source
  station's strip start dates and exterior film coefficient.
* Buildings: training-sized ones (the shipped buildings' footprint and room
  ranges, narrower than the generator defaults) and two default-range ones.
* Scenarios under smart_buildings/wide_scenarios/: train = 12 small buildings x 2
  training climates + 4 training buildings x 2 unseen strips; holdout = 4 small
  buildings x 3 holdout climates + 2 large buildings x 2 holdout climates.

Tuning sees neither the holdout buildings nor the holdout climates.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

KIT = Path("/data/smartbuild/starter-kit")
W = KIT / "smart_buildings" / "weather"
OUT = "smart_buildings/wide_scenarios"

# name: (source station, offset K, diurnal swing scale, role)
CLIMATES = {
    "syn_desert": ("phoenix", 4.0, 1.15, "train"),        # Riyadh-like summer peaks near 50 C
    "syn_subarctic": ("minneapolis", -5.0, 1.0, "train"),
    "syn_tropic": ("singapore", 2.5, 1.3, "holdout"),
    "syn_highland": ("denver", -3.0, 1.4, "holdout"),
    "syn_marine": ("atlanta", -4.0, 0.5, "holdout"),
}
TRAIN_CLIMATES = ["fairbanks", "denver", "atlanta", "phoenix", "singapore", "minneapolis",
                  "syn_desert", "syn_subarctic"]
HOLDOUT_CLIMATES = ["syn_tropic", "syn_highland", "syn_marine"]
# The shipped training buildings span these; the generator defaults are larger.
SMALL_RANGES = dict(width=(330, 440), height=(180, 310), min_room_size=(25, 36))


def make_climates() -> None:
    for name, (src, offset, scale, _) in CLIMATES.items():
        df = pd.read_csv(W / "csv" / f"{src}.csv", index_col=0)
        t = df["TempC"].astype(float)
        mean = t.rolling(24, center=True, min_periods=1).mean()
        df["TempC"] = (mean + offset + scale * (t - mean)).round(2)
        df["TempF"] = (df["TempC"] * 9 / 5 + 32).round(2)
        df["StationName"] = name
        df.to_csv(W / "csv" / f"{name}.csv")
        for n in range(1, 9):
            preset = yaml.safe_load((W / f"{src}_{n}.yaml").read_text())
            preset.update(name=f"{name}_{n}", weather_csv=f"csv/{name}.csv")
            (W / f"{name}_{n}.yaml").write_text(yaml.safe_dump(preset, sort_keys=False))
        print(f"climate {name}: {src} {offset:+.1f} K x{scale}, range {df['TempC'].min():.1f}..{df['TempC'].max():.1f} C")


def make_small_buildings(start_seed: int, count: int, out_dir: str) -> None:
    sys.path.insert(0, str(KIT))
    from smart_buildings import run_floor_generator as rfg

    rfg.DEFAULT_RANGES.update(SMALL_RANGES)
    rfg.main(["--start-seed", str(start_seed), "--count", str(count), "--output-dir", out_dir, "--no-pdf"])


def gen(building: str, weathers: list[str]) -> tuple[str, bool]:
    r = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator", "--buildings", building,
                        "--weathers", *weathers, "--output-dir", OUT], cwd=KIT, capture_output=True, text=True)
    return building, r.returncode == 0


def valid_buildings(paths: list[Path]) -> list[Path]:
    """Keep buildings the scenario generator accepts.

    Narrow room ranges make the generator reject more seeds (a zone under its
    20 m2 minimum), so a pool is generated and each is tried once into /tmp.
    """
    def ok(p: Path) -> bool:
        r = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator", "--buildings", str(p),
                            "--weathers", "atlanta_1", "--output-dir", f"/tmp/wide_check/{p.stem}"],
                           cwd=KIT, capture_output=True, text=True)
        return r.returncode == 0

    with ThreadPoolExecutor(16) as pool:
        flags = list(pool.map(ok, paths))
    subprocess.run(["rm", "-rf", "/tmp/wide_check"])
    return [p for p, f in zip(paths, flags) if f]


def main() -> None:
    os.chdir(KIT)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(["rm", "-rf", OUT, "smart_buildings/buildings_small", "smart_buildings/buildings_large"])
    make_climates()
    make_small_buildings(700, 60, "smart_buildings/buildings_small")
    subprocess.run([sys.executable, "-m", "smart_buildings.run_floor_generator", "--seeds", "502", "511",
                    "--output-dir", "smart_buildings/buildings_large", "--no-pdf"], check=True,
                   stdout=subprocess.DEVNULL)

    jobs = []
    small = valid_buildings(sorted((KIT / "smart_buildings/buildings_small").glob("building_*.yaml")))
    print(f"{len(small)} of 60 small buildings pass the generator's checks")
    for i, b in enumerate(small[:12]):  # train: two training climates each
        n = i % 8 + 1
        jobs.append((str(b), [f"{TRAIN_CLIMATES[i % 8]}_{n}", f"{TRAIN_CLIMATES[(i + 3) % 8]}_{(n + 3) % 8 + 1}"]))
    for i, b in enumerate(small[12:16]):  # holdout: all three holdout climates
        jobs.append((str(b), [f"{c}_{(i + k) % 8 + 1}" for k, c in enumerate(HOLDOUT_CLIMATES)]))
    for i, b in enumerate(sorted((KIT / "smart_buildings/buildings_large").glob("building_*.yaml"))):
        jobs.append((str(b), [f"{HOLDOUT_CLIMATES[i]}_{i + 5}", f"{HOLDOUT_CLIMATES[i + 1]}_{i + 6}"]))
    for b, n in [("building_train_1", 6), ("building_train_4", 7), ("building_train_7", 2), ("building_train_8", 3)]:
        jobs.append((b, [f"fairbanks_{n}", f"phoenix_{n}"]))
    with ThreadPoolExecutor(16) as pool:
        for b, ok in pool.map(lambda j: gen(*j), jobs):
            if not ok:
                print("FAILED", b)
    n = sum(1 for _ in (KIT / OUT).glob("*/*/config.gin"))
    print(f"{n} scenarios in {OUT}")
    print("train:", " ".join(p.stem for p in small[:12]))
    print("holdout:", " ".join(p.stem for p in small[12:16]))


if __name__ == "__main__":
    main()
