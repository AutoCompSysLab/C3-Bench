"""Evaluate change-caption predictions with an LLM judge.

The judge model scores each prediction against its reference caption on four
criteria (correctness, specificity, relevance, fluency) plus a single
``final_score``, all on a 1-10 scale. Predictions produced by
:mod:`c3_bench.infer` can be fed in directly.

Example
-------
    python -m c3_bench.eval \\
        --judge Qwen/Qwen2.5-VL-3B-Instruct \\
        --model-name Qwen2.5-VL-3B-Instruct \\
        --result-dir result --out-dir eval_outputs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev

from tqdm.auto import tqdm

from . import prompts
from .io_utils import model_basename
from .models import DEFAULT_GEN_KWARGS, build_inputs, generate, load_model_and_processor, set_seed

SCORE_TAGS = ("correctness", "specificity", "relevance", "fluency", "final_score")


def _extract_prediction(raw: str) -> str:
    """Pull the text inside ``<output>...</output>``; fall back to the raw string."""
    match = re.search(r"<output>\s*(.*?)\s*</output>", raw, re.DOTALL)
    return match.group(1) if match else raw


def _parse_score(text: str, tag: str) -> int | None:
    """Parse the integer score inside ``<tag>...</tag>``; ``None`` if absent/empty.

    Tolerates formats such as ``"8/10"`` by taking the leading integer.
    """
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if not match:
        return None
    leading = re.split(r"[^0-9]+", match.group(1).strip())[0]
    return int(leading) if leading else None


def _summary(values: list[int]) -> tuple[float | None, float | None]:
    """Return ``(mean, std)`` for a list of scores (std is 0.0 for a single run)."""
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _first_or_mean(values: list[int], mean_value: float | None) -> int | None:
    """Representative score: first run if available, else the rounded mean."""
    if values:
        return values[0]
    return int(round(mean_value)) if mean_value is not None else None


def run_eval(
    *,
    judge_id: str,
    model_name: str,
    predictions_path: Path,
    out_dir: Path,
    num_repeats: int,
    max_new_tokens: int,
    attn_impl: str | None,
) -> Path:
    """Score every prediction and write per-item results to ``out_dir``.

    Returns:
        The path of the written evaluation JSON file.
    """
    if num_repeats < 1:
        raise ValueError(f"--num-repeats must be >= 1, got {num_repeats}.")

    set_seed(42)

    print(f"[INFO] Loading judge model: {judge_id}")
    model, processor = load_model_and_processor(judge_id, attn_impl=attn_impl)

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"[INFO] Loaded {len(predictions)} predictions from {predictions_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"evaluation_{model_name}.json"
    error_path = out_dir / f"error_{model_name}.json"

    gen_kwargs = {**DEFAULT_GEN_KWARGS, "max_new_tokens": max_new_tokens}

    results: list[dict] = []
    errors: list[dict] = []

    for item in tqdm(predictions, desc=f"judge:{model_basename(judge_id)}", dynamic_ncols=True):
        prediction = _extract_prediction(item["prediction"])
        reference = item["gt_captions"][0]

        scores: dict[str, list[int]] = {tag: [] for tag in SCORE_TAGS}
        last_judge_text: str | None = None
        try:
            for _ in range(num_repeats):
                eval_prompt = prompts.build_eval_prompt(prediction, reference)
                model_inputs = build_inputs(model, processor, "eval", eval_prompt)
                last_judge_text = generate(model, processor, model_inputs, gen_kwargs)
                for tag in SCORE_TAGS:
                    value = _parse_score(last_judge_text, tag)
                    if value is not None:
                        scores[tag].append(value)
        except Exception as exc:  # noqa: BLE001 - per-item failures must not abort the run
            print(f"[error] index={item.get('index')}: {exc}")
            errors.append({
                "index": item.get("index"),
                "image_t0_path": item.get("image_t0_path"),
                "image_t1_path": item.get("image_t1_path"),
                "last_judge_text": last_judge_text,
                "error": str(exc),
            })

        summaries = {tag: _summary(scores[tag]) for tag in SCORE_TAGS}

        record: dict = {
            "domain": item["domain"],
            "topic": item["topic"],
            "index": item["index"],
            "judge": judge_id,
            "image_t0_path": item["image_t0_path"],
            "image_t1_path": item["image_t1_path"],
        }
        # Representative single value (kept for backward compatibility with
        # downstream analysis scripts), then per-run lists, then mean/std.
        for tag in SCORE_TAGS:
            record[tag] = _first_or_mean(scores[tag], summaries[tag][0])
        for tag in SCORE_TAGS:
            record[f"{tag}_all"] = scores[tag]
        for tag in SCORE_TAGS:
            m, sd = summaries[tag]
            record[f"{tag}_mean"] = m
            record[f"{tag}_std"] = sd

        results.append(record)

        # Checkpoint after every item.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if errors:
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"[saved] {out_path}")
    if errors:
        print(f"[errors] {len(errors)} samples logged to {error_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score change-caption predictions with an LLM judge.")
    p.add_argument("--judge", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HuggingFace model id of the judge.")
    p.add_argument("--model-name", default="Qwen2.5-VL-3B-Instruct",
                   help="Name of the evaluated model; used to derive default file names.")
    p.add_argument("--predictions", type=Path, default=None,
                   help="Predictions JSON to evaluate. "
                        "Default: <result-dir>/prediction_<model-name>.json")
    p.add_argument("--result-dir", type=Path, default=Path("result"),
                   help="Directory holding the predictions JSON (used for the default --predictions).")
    p.add_argument("--out-dir", type=Path, default=Path("eval_outputs"),
                   help="Directory for the evaluation and error JSON files.")
    p.add_argument("--num-repeats", type=int, default=1,
                   help="Number of judge passes per item (scores are averaged).")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Maximum number of generated tokens for the judge.")
    p.add_argument("--attn-impl", default=None, choices=[None, "flash_attention_2"],
                   help="Attention implementation (supported environments only).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    predictions_path = args.predictions or (args.result_dir / f"prediction_{args.model_name}.json")
    run_eval(
        judge_id=args.judge,
        model_name=args.model_name,
        predictions_path=predictions_path,
        out_dir=args.out_dir,
        num_repeats=args.num_repeats,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.attn_impl,
    )


if __name__ == "__main__":
    main()
