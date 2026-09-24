#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python_bin="${PYTHON:-/mnt/e/repos/GGGT/.venv/bin/python}"
exec "$python_bin" -u tools/evaluate_genimage.py \
  --checkpoint outputs/genimage_sd14_fixed_dinov3l_lora_r8_20ep/final.pt \
  --output outputs/genimage_sd14_fixed_dinov3l_lora_r8_20ep/genimage_evaluation \
  --batch-size 128 "$@"
