#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python_bin="${PYTHON:-/mnt/e/repos/GGGT/.venv/bin/python}"
exec "$python_bin" -u tools/train.py \
  --config configs/genimage_sd14_fixed_dinov3_lora_mlp224_concat20.yaml "$@"
