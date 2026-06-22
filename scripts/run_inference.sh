#!/usr/bin/env bash
# Generate change captions for image pairs with a VLM.
# Run from the repository root so `python -m c3_bench.infer` resolves the package.
set -euo pipefail

# Optional: point HuggingFace caches at a shared location.
# export HF_HOME=/path/to/hf_cache

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
JSON_PATH="${JSON_PATH:-data/items.json}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"
CASE="${CASE:-no-tag-forward}"
OUT_DIR="${OUT_DIR:-outputs}"

python -m c3_bench.infer \
  --project-root "${PROJECT_ROOT}" \
  --json-path "${JSON_PATH}" \
  --model-id "${MODEL_ID}" \
  --case "${CASE}" \
  --criteria \
  --out-dir "${OUT_DIR}" \
  --max-new-tokens 2048
