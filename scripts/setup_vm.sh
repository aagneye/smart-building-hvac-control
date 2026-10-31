#!/usr/bin/env bash
# Set up, or restore after a reboot, the VM workspace.
#
# The workspace lives on the persistent root disk at /home/azureuser/smartbuild
# and is linked as /data/smartbuild. Do not move it to /mnt: that is the Azure
# temporary disk and is wiped on every reboot (it was, on 2026-10-31).
#
# Run from your machine:
#   ssh azureuser@20.51.160.183 'bash -s' < scripts/setup_vm.sh   # workspace + link + envs
#   bash scripts/sync.sh                                           # push our code
#   ssh azureuser@20.51.160.183 'bash /data/smartbuild/scripts/setup_vm.sh'  # installs our package too
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
WS=/home/azureuser/smartbuild
KIT_COMMIT=95cea5b   # starter kit version our results were measured on

mkdir -p "$WS/logs" "$WS/runs" "$WS/kaggle"
if [ "$(readlink /data/smartbuild || true)" != "$WS" ]; then
  sudo ln -sfn "$WS" /data/smartbuild
fi
cd "$WS"

if [ ! -d starter-kit ]; then
  git clone -q https://github.com/smart-buildings-2026/starter-kit.git
fi
git -C starter-kit checkout -q "$KIT_COMMIT"

# Simulator environment.
[ -d .venv ] || uv venv -q -p 3.12 .venv
uv pip install -q -p .venv -e starter-kit cma pyarrow
if [ -f pyproject.toml ]; then
  uv pip install -q -p .venv -e .
fi

# Separate environment for the real Kaggle gateway test: it needs grpc tools,
# which would otherwise pull a protobuf the simulator cannot use.
[ -d .venv-gw ] || uv venv -q -p 3.12 .venv-gw
uv pip install -q -p .venv-gw numpy==2.0.2 numba==0.60.0 pandas==2.3.3 matplotlib==3.10.0 \
  gymnasium==1.2.0 gin-config==0.5.0 protobuf==5.29.5 pytz==2025.2 holidays==0.95 scipy==1.16.3 \
  pyyaml==6.0.3 absl-py==1.4.0 importlib_resources==7.1.0 pillow==11.3.0 \
  opencv-python-headless==4.13.0.92 imageio==2.37.3 tqdm==4.67.3 seaborn==0.13.2 plotly==5.24.1 \
  scikit-learn==1.6.1 grpcio==1.70.0 grpcio-tools==1.70.0 polars pyarrow pydantic \
  opentelemetry-api opentelemetry-sdk

# Competition data (gateway + bundled simulator + example scenario), 2.5 MB zip.
if [ ! -d kaggle/comp/data ]; then
  mkdir -p kaggle/comp
  (cd kaggle/comp && kaggle competitions download -c neurips-2026-smart-buildings -p . \
    && mkdir -p data && cd data && unzip -q ../neurips-2026-smart-buildings.zip)
fi

echo "workspace ready at $WS ($(du -sh "$WS" | cut -f1))"
