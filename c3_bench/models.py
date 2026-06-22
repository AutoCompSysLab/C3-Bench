"""Vision-language model loading, input building, and generation.

A single :func:`load_model_and_processor` entry point handles every supported
model family, replacing the per-family ``if/elif`` blocks that were previously
duplicated across the inference and evaluation scripts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ``AutoModelForVision2Seq`` was renamed to ``AutoModelForImageTextToText`` in
# transformers v5. Prefer the modern name and fall back to the legacy one so the
# package works across versions.
try:
    from transformers import AutoModelForImageTextToText as _AutoVLM
except ImportError:  # transformers < 4.46
    from transformers import AutoModelForVision2Seq as _AutoVLM

from .io_utils import model_basename

# Default decoding configuration: deterministic greedy decoding.
DEFAULT_GEN_KWARGS: dict[str, Any] = {
    "max_new_tokens": 2048,
    "num_beams": 1,
    "repetition_penalty": 1.0,
    "do_sample": False,
}

# Message ``case`` identifiers understood by :func:`build_inputs`.
IMAGE_CASES = ("no-tag-forward", "no-tag-backward", "tag-forward", "tag-backward")
EVAL_CASE = "eval"
VALID_CASES = IMAGE_CASES + (EVAL_CASE,)

# Sentinel meaning "bfloat16 on GPU, float32 on CPU".
_BF16_OR_FP32 = object()


@dataclass(frozen=True)
class _ModelSpec:
    loader: type
    torch_dtype: Any            # "auto", a torch.dtype, or the _BF16_OR_FP32 sentinel
    trust_remote_code: bool
    supports_attn_impl: bool


# Specs are matched by substring against the model basename, in declaration order.
# (InternVL3_5- precedes InternVL3- so the more specific name wins.)
_MODEL_SPECS: tuple[tuple[str, _ModelSpec], ...] = (
    ("InternVL3_5-", _ModelSpec(_AutoVLM, _BF16_OR_FP32, False, False)),
    ("InternVL3-", _ModelSpec(_AutoVLM, _BF16_OR_FP32, False, False)),
    ("Qwen3-", _ModelSpec(_AutoVLM, "auto", True, True)),
    ("Qwen2.5-", _ModelSpec(Qwen2_5_VLForConditionalGeneration, "auto", True, True)),
    ("Llama-3.2", _ModelSpec(_AutoVLM, "auto", True, True)),
    ("llava-", _ModelSpec(_AutoVLM, torch.float16, True, True)),
)

#: Model-family prefixes recognised by :func:`load_model_and_processor`.
SUPPORTED_MODEL_FAMILIES = tuple(prefix for prefix, _ in _MODEL_SPECS)


def set_seed(seed: int = 42) -> None:
    """Seed Python/NumPy/Torch RNGs.

    Determinism is intentionally relaxed (``cudnn.benchmark = True``) to favour
    throughput, matching the original benchmark configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _resolve_spec(model_id: str) -> _ModelSpec | None:
    name = model_basename(model_id)
    for prefix, spec in _MODEL_SPECS:
        if prefix in name:
            return spec
    return None


def load_model_and_processor(model_id: str, attn_impl: str | None = None):
    """Load a supported VLM and its processor with ``device_map="auto"``.

    Args:
        model_id: HuggingFace model id, e.g. ``"Qwen/Qwen2.5-VL-3B-Instruct"``.
        attn_impl: optional attention implementation (e.g. ``"flash_attention_2"``);
            only applied to families that support it.

    Returns:
        A ``(model, processor)`` tuple with the model placed in eval mode and
        sharded across available GPUs.

    Raises:
        ValueError: if the model family is not supported.
    """
    spec = _resolve_spec(model_id)
    if spec is None:
        raise ValueError(
            f"Unsupported model '{model_id}'. "
            f"Supported families: {', '.join(SUPPORTED_MODEL_FAMILIES)}."
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if spec.torch_dtype is _BF16_OR_FP32:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        torch_dtype = spec.torch_dtype

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "device_map": "auto"}
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if spec.supports_attn_impl and attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = spec.loader.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, processor


def build_messages(case: str, prompt: str, img0=None, img1=None) -> list[dict]:
    """Build the chat ``messages`` payload for a given ``case``.

    Image cases place the two images in forward (``t0``, ``t1``) or backward
    (``t1``, ``t0``) order, optionally preceded by ``[image_t0]``/``[image_t1]``
    text tags. The ``"eval"`` case is text-only (used by the LLM judge).

    Args:
        case: one of :data:`VALID_CASES`.
        prompt: the textual instruction.
        img0, img1: image paths (or PIL images); required for image cases.
    """
    if case == EVAL_CASE:
        content = [{"type": "text", "text": prompt}]
    elif case == "no-tag-forward":
        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "image": img0},
            {"type": "image", "image": img1},
        ]
    elif case == "tag-forward":
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "[image_t0]"},
            {"type": "image", "image": img0},
            {"type": "text", "text": "[image_t1]"},
            {"type": "image", "image": img1},
        ]
    elif case == "no-tag-backward":
        content = [
            {"type": "text", "text": prompt},
            {"type": "image", "image": img1},
            {"type": "image", "image": img0},
        ]
    elif case == "tag-backward":
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "[image_t1]"},
            {"type": "image", "image": img1},
            {"type": "text", "text": "[image_t0]"},
            {"type": "image", "image": img0},
        ]
    else:
        raise ValueError(f"Unknown case '{case}'. Expected one of {VALID_CASES}.")
    return [{"role": "user", "content": content}]


def build_inputs(model, processor, case: str, prompt: str, img0=None, img1=None):
    """Build tokenised chat inputs for a given ``case``, moved to ``model.device``.

    See :func:`build_messages` for the per-case message layout.
    """
    messages = build_messages(case, prompt, img0, img1)
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


@torch.no_grad()
def generate(model, processor, model_inputs, gen_kwargs: dict[str, Any] | None = None) -> str:
    """Run generation and return the decoded completion (prompt stripped)."""
    gen_kwargs = DEFAULT_GEN_KWARGS if gen_kwargs is None else gen_kwargs
    generated_ids = model.generate(**model_inputs, **gen_kwargs)
    if "input_ids" in model_inputs:
        generated_ids = generated_ids[:, model_inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
