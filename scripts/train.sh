#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
config="configs/genimage_sd14_fixed.yaml"
output=""
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == "--config" ]]; then j=$((i+1)); config="${!j}"; fi
  if [[ "${!i}" == "--output" ]]; then j=$((i+1)); output="${!j}"; fi
done
if [[ -z "$output" ]]; then
  output=$("${PYTHON:-.venv/bin/python}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output"])' "$config")
fi
mkdir -p "$output"
"${PYTHON:-.venv/bin/python}" -u tools/train.py "$@" 2>&1 | tee -a "$output/train.log"
