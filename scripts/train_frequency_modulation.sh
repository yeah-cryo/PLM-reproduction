#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

exec bash scripts/train.sh \
  --config configs/genimage_sd14_frequency_modulation_20.yaml "$@"
