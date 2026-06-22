"""Run a vision-language model over image pairs to generate change captions.

Example
-------
    python -m c3_bench.infer \\
        --project-root . \\
        --json-path data/items.json \\
        --model-id Qwen/Qwen2.5-VL-3B-Instruct \\
        --case no-tag-forward --criteria --out-dir outputs
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from tqdm.auto import tqdm

from . import prompts
from .dist_utils import init_distributed, is_main_process
from .io_utils import infer_domain_topic, load_items, model_basename, resolve_image
from .models import DEFAULT_GEN_KWARGS, IMAGE_CASES, build_inputs, generate, load_model_and_processor, set_seed


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_retry_indices(path: Path) -> set[int]:
    """Read a Python-literal list of item indices to re-run (e.g. ``[3, 7, 12]``)."""
    content = Path(path).read_text(encoding="utf-8").strip()
    return {i for i in (_as_int(x) for x in ast.literal_eval(content)) if i is not None}


def run_inference(
    *,
    project_root: Path,
    json_path: Path,
    model_id: str,
    out_dir: Path,
    case: str,
    use_criteria: bool,
    max_new_tokens: int,
    attn_impl: str | None,
    parallel_mode: str,
    retry_error_file: Path | None = None,
) -> Path:
    """Generate change captions for every item and write them to ``out_dir``.

    Returns:
        The path of the written predictions JSON file.
    """
    if case not in IMAGE_CASES:
        raise ValueError(f"--case must be one of {IMAGE_CASES}, got {case!r}.")

    rank, world_size, local_rank = (0, 1, 0)
    use_ddp = parallel_mode == "ddp"
    if use_ddp:
        rank, world_size, local_rank = init_distributed()

    if is_main_process(rank):
        print(f"[INFO] Parallel mode: {parallel_mode} | world_size={world_size} | local_rank={local_rank}")

    set_seed(42)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = model_basename(model_id)
    stem = f"prediction_{base}_{case}_{use_criteria}"
    out_path = out_dir / (f"{stem}.rank{rank}.json" if use_ddp else f"{stem}.json")

    if is_main_process(rank):
        print(f"[INFO] Loading model: {model_id}")
    model, processor = load_model_and_processor(model_id, attn_impl=attn_impl)

    items = load_items(json_path)
    if retry_error_file is not None:
        indices = _load_retry_indices(retry_error_file)
        items = [it for it in items if _as_int(it.get("index")) in indices]
        if is_main_process(rank):
            print(f"[INFO] Retry filter: {len(indices)} indices -> {len(items)} items")

    if is_main_process(rank):
        print(f"[INFO] Loaded {len(items)} items from {json_path}")

    shard = items[rank::world_size] if use_ddp else items
    if use_ddp and is_main_process(rank):
        print(f"[INFO] Sharding {len(items)} items -> ~{len(shard)} per rank")

    gen_kwargs = {**DEFAULT_GEN_KWARGS, "max_new_tokens": max_new_tokens}
    tagged = "no-tag" not in case

    preds: list[dict] = []
    ok_cnt = skip_cnt = 0
    show_progress = is_main_process(rank) or not use_ddp
    iterator = enumerate(shard, 1)
    if show_progress:
        iterator = tqdm(iterator, total=len(shard), desc=base, dynamic_ncols=True)

    for i, item in iterator:
        try:
            domain, topic = infer_domain_topic(item)
            prompt = prompts.build_inference_prompt(domain, topic, use_criteria=use_criteria, tagged=tagged)

            gt_captions = item.get("captions", [])
            if not isinstance(gt_captions, list):
                gt_captions = [str(gt_captions)]

            img0 = resolve_image(item["image_t0_path"], project_root)
            img1 = resolve_image(item["image_t1_path"], project_root)
            if not img0.exists():
                raise FileNotFoundError(f"image_t0 not found: {img0}")
            if not img1.exists():
                raise FileNotFoundError(f"image_t1 not found: {img1}")

            model_inputs = build_inputs(model, processor, case, prompt, str(img0), str(img1))
            output_text = generate(model, processor, model_inputs, gen_kwargs)

            preds.append({
                "model_id": model_id,
                "prediction": output_text,
                "index": item.get("index"),
                "domain": domain,
                "topic": topic,
                "image_t0_path": str(img0),
                "image_t1_path": str(img1),
                "prompt_used": prompt,
                "gt_captions": gt_captions,
            })
            ok_cnt += 1
            # Checkpoint after every item so a crash never loses completed work.
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001 - per-item failures must not abort the run
            skip_cnt += 1
            if show_progress:
                prefix = f"(rank {rank}) " if use_ddp else ""
                tqdm.write(f"[WARN] {prefix}skipped item {i}: {exc}")

        if show_progress:
            iterator.set_postfix(ok=ok_cnt, skip=skip_cnt)

    if is_main_process(rank):
        print(f"[INFO] Saved {ok_cnt} predictions to {out_path} (skipped {skip_cnt}).")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate change captions for image pairs with a VLM.")
    p.add_argument("--project-root", type=Path, required=True,
                   help="Root used to resolve relative image paths in the items file.")
    p.add_argument("--json-path", type=Path, required=True,
                   help="Path to the items JSON (list of records with image_t0_path/image_t1_path).")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HuggingFace model id to run.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"),
                   help="Directory for the predictions JSON.")
    p.add_argument("--case", default="no-tag-forward", choices=list(IMAGE_CASES),
                   help="Image ordering / tagging scheme.")
    p.add_argument("--criteria", action="store_true",
                   help="Use the criteria-guided template instead of the naive one.")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Maximum number of generated tokens.")
    p.add_argument("--attn-impl", default=None, choices=[None, "flash_attention_2"],
                   help="Attention implementation (supported environments only).")
    p.add_argument("--parallel-mode", default="hf-auto", choices=["hf-auto", "ddp"],
                   help="'hf-auto': single process, device_map='auto'. 'ddp': torchrun-based sharding.")
    p.add_argument("--retry-error-file", type=Path, default=None,
                   help="Optional file containing a Python-literal list of indices to re-run.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_inference(
        project_root=args.project_root,
        json_path=args.json_path,
        model_id=args.model_id,
        out_dir=args.out_dir,
        case=args.case,
        use_criteria=args.criteria,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.attn_impl,
        parallel_mode=args.parallel_mode,
        retry_error_file=args.retry_error_file,
    )


if __name__ == "__main__":
    main()
