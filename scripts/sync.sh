#!/usr/bin/env bash
# Push local code to the VM work folder. Usage: bash scripts/sync.sh
set -euo pipefail
HOST=${SB_HOST:-azureuser@20.51.160.183}
cd "$(dirname "$0")/.."
tar czf - src scripts params pyproject.toml 2>/dev/null | ssh -o BatchMode=yes "$HOST" 'cd /data/smartbuild && tar xzf -'
echo "synced to $HOST:/data/smartbuild"
