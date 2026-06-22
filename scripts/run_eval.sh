#!/usr/bin/env bash
# Score change-caption predictions with an LLM judge.
# Run from the repository root so `python -m c3_bench.eval` resolves the package.
set -euo pipefail

# Optional: point HuggingFace caches at a shared location.
# export HF_HOME=/path/to/hf_cache

JUDGE="${JUDGE:-Qwen/Qwen2.5-VL-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-3B-Instruct}"
RESULT_DIR="${RESULT_DIR:-result}"
OUT_DIR="${OUT_DIR:-eval_outputs}"

# By default the predictions file is <RESULT_DIR>/prediction_<MODEL_NAME>.json.
python -m c3_bench.eval \
  --judge "${JUDGE}" \
  --model-name "${MODEL_NAME}" \
  --result-dir "${RESULT_DIR}" \
  --out-dir "${OUT_DIR}" \
  --num-repeats 1 \
  --max-new-tokens 2048
