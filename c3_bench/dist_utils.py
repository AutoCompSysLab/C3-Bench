"""Distributed-execution helpers (single-process and torchrun/DDP)."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Initialise the process group when launched via ``torchrun``.

    Falls back to a single-process configuration when the distributed
    environment variables (``RANK``/``WORLD_SIZE``) are not set.

    Returns:
        A ``(rank, world_size, local_rank)`` tuple.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Generous timeout so long generation steps do not trip the collective.
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def barrier() -> None:
    """Synchronise all ranks if a process group is active; no-op otherwise."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int) -> bool:
    """Return ``True`` for the rank-0 (main) process."""
    return rank == 0
