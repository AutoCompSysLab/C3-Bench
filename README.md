# C3-Bench

**C**riteria-guided **C**hange **C**aptioning **Bench**mark for vision-language models (VLMs).

C3-Bench evaluates how well a VLM can describe the visual change between two
semantically related images (`image_t0` -> `image_t1`). It ships:

- a **criteria-guided prompting** scheme with per-domain *change criteria* that
  specify which changes to report and which variations (lighting, viewpoint,
  compression, …) to ignore;
- an **inference** pipeline that runs a VLM over image pairs and saves captions;
- an **LLM-as-judge evaluation** pipeline that scores predictions against
  reference captions on four axes plus an overall score.

Supported domains: `natural_scenes`, `anomalies` (MVTec/VisA-style defects),
`image_editing`, and `satellite_imagery`.

## Repository layout

```
c3_bench/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── c3_bench/                # the Python package
│   ├── __init__.py
│   ├── prompts.py           # change criteria + inference/judge templates
│   ├── models.py            # unified VLM loading, input building, generation
│   ├── io_utils.py          # JSON / path / domain-topic helpers
│   ├── dist_utils.py        # single-process & torchrun (DDP) helpers
│   ├── infer.py             # entry point: run inference  (python -m c3_bench.infer)
│   └── eval.py              # entry point: LLM-judge eval (python -m c3_bench.eval)
└── scripts/
    ├── run_inference.sh
    └── run_eval.sh
```

## Installation

```bash
git clone <your-repo-url> c3_bench
cd c3_bench
pip install -r requirements.txt
```

Requires Python 3.10+. Model weights are downloaded from the HuggingFace Hub on
first use; set `HF_HOME` to control the cache location:

```bash
export HF_HOME=/path/to/hf_cache
```

### Supported model families

The loader recognises the following id prefixes and selects the appropriate
HuggingFace class automatically:

`Qwen3-`, `Qwen2.5-`, `InternVL3-`, `InternVL3_5-`, `Llama-3.2`, `llava-`.

## Data format

The items file passed to inference is a JSON list of records:

```json
[
  {
    "index": 0,
    "domain": "natural_scenes",
    "topic": "meeting_room",
    "image_t0_path": "data/natural_scenes/meeting_room/0_t0.jpg",
    "image_t1_path": "data/natural_scenes/meeting_room/0_t1.jpg",
    "captions": ["A laptop has been placed on the table."]
  }
]
```

- `domain` / `topic` select the change criteria. If omitted, they are inferred
  from image paths of the form `data/<domain>/<topic>/...`.
- Relative `image_*_path` values are resolved against `--project-root`.
- `captions` holds the reference caption(s) used later by the judge.

## Usage

> Run the commands from the repository root so that `python -m c3_bench.*`
> resolves the package.

### 1. Inference

```bash
python -m c3_bench.infer \
    --project-root . \
    --json-path data/items.json \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct \
    --case no-tag-forward \
    --criteria \
    --out-dir outputs
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--case` | `no-tag-forward` | Image order/tagging: `{no-tag,tag}-{forward,backward}`. |
| `--criteria` | off | Use the criteria-guided template instead of the naive one. |
| `--max-new-tokens` | `2048` | Generation length cap. |
| `--attn-impl` | `None` | e.g. `flash_attention_2` where available. |
| `--parallel-mode` | `hf-auto` | `hf-auto` (single process, `device_map="auto"`) or `ddp`. |
| `--retry-error-file` | `None` | Re-run only a Python-literal list of `index` values. |

Output: `outputs/prediction_<model>_<case>_<criteria>.json`, a list of records
each containing the `prediction`, the exact `prompt_used`, and the reference
`gt_captions`. The file is rewritten after every item, so an interrupted run
keeps its completed work.

**Image ordering cases**

- `forward` presents `image_t0` then `image_t1`; `backward` reverses them.
- `tag` prefixes each image with an `[image_t0]` / `[image_t1]` text marker;
  `no-tag` leaves the order implicit.

### 2. Evaluation (LLM judge)

```bash
python -m c3_bench.eval \
    --judge Qwen/Qwen2.5-VL-3B-Instruct \
    --model-name Qwen2.5-VL-3B-Instruct \
    --result-dir result \
    --out-dir eval_outputs \
    --num-repeats 1
```

The judge reads `result/prediction_<model-name>.json` (override with
`--predictions`) and scores each item on **correctness**, **specificity**,
**relevance**, and **fluency** (1-10), plus a single human-aligned
**final_score**. Results are written to
`eval_outputs/evaluation_<model-name>.json`; per-item failures are logged to
`eval_outputs/error_<model-name>.json`.

With `--num-repeats > 1` the judge is queried multiple times per item and the
output additionally reports `*_all`, `*_mean`, and `*_std` for each metric.

> The `--predictions` produced by inference are named `prediction_<...>.json`
> in `outputs/`; copy or point `--predictions` at the file you want to judge.

### Convenience scripts

`scripts/run_inference.sh` and `scripts/run_eval.sh` wrap the commands above and
read their settings from environment variables (`MODEL_ID`, `JSON_PATH`, `CASE`,
`JUDGE`, `MODEL_NAME`, …).

## Customising change criteria

All criteria and templates live in [`c3_bench/prompts.py`](c3_bench/prompts.py).
Each is a module-level constant named `<domain>[_<topic>]_change_criteria`; they
are auto-collected into `CHANGE_CRITERIA` and resolved per item with a
`domain_topic` -> `domain` -> `generic` fallback. To add a new domain or topic,
define a new `*_change_criteria` string — no other code changes are required.

## License

Released under the [MIT License](LICENSE).
