#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

spacing="${1:-16}"
if [[ "$spacing" != "4" && "$spacing" != "16" && "$spacing" != "64" ]]; then
  echo "Usage: bash scripts/train_smooth_mapping.sh {4|16|64} [additional train arguments]" >&2
  exit 2
fi
shift || true

exec bash scripts/train.sh --config "configs/genimage_sd14_smooth_h${spacing}.yaml" "$@"
