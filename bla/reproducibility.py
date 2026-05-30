"""Single-call RNG seeding for BLA training + eval scripts.

Most scripts in this repo set torch + numpy seeds individually. That misses
the `random` module, CUDA, MPS, and `PYTHONHASHSEED`. The DR3 / "execution
stochasticity" finding (memory: feedback_execution_stochasticity_not_metric)
showed that residual run-to-run variance on Lift was env/MuJoCo RNG, not the
metric. This module collapses seed handling to one call so every entry
point seeds the same set of RNGs the same way.

Usage:

    from bla.reproducibility import set_all_seeds
    set_all_seeds(args.seed)

    # If you need a DataLoader-local generator alongside global seeding:
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    loader = DataLoader(..., generator=g, worker_init_fn=worker_init_fn)

What this does NOT do:
  - Force cuDNN deterministic mode (would slow training; opt in via
    `set_all_seeds(seed, deterministic_cudnn=True)` when reproducing a
    published gate value).
  - Seed simulator RNG (MuJoCo, Robosuite). Those have their own
    `env.seed(...)` and `np.random.seed(...)` paths; call them from the
    env constructor. This module covers the in-process RNGs only.
"""
from __future__ import annotations

import os
import random
from typing import Optional


def set_all_seeds(
    seed: int,
    *,
    deterministic_cudnn: bool = False,
    pythonhashseed: bool = True,
) -> int:
    """Seed Python `random`, NumPy, and PyTorch (CPU + CUDA + MPS).

    Args:
      seed:                int seed shared by all RNGs.
      deterministic_cudnn: if True, also set cudnn.deterministic=True
                           and cudnn.benchmark=False. Slows training;
                           use only when reproducing a pinned metric.
      pythonhashseed:      if True (default), set PYTHONHASHSEED in the
                           current process env. Note: this only affects
                           SUBPROCESSES; the current interpreter's hash
                           seed was already chosen at startup. For the
                           strictest reproduction, set PYTHONHASHSEED in
                           your shell before running python.

    Returns:
      The seed value, for convenience: `seed = set_all_seeds(args.seed)`.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    if pythonhashseed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Per-DataLoader-worker seeding. Pass as `worker_init_fn=` arg.

    Combines the base PyTorch worker seed with the worker_id so each
    worker uses a distinct but deterministic stream.
    """
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_torch_generator(seed: int, device: str = "cpu"):
    """Build a torch.Generator pinned to (seed, device).

    Use for DataLoader shuffling or other call sites that take a
    `generator=` kwarg, so loader-side randomness is independent of
    `torch.manual_seed` calls elsewhere in the process.
    """
    import torch
    return torch.Generator(device=device).manual_seed(int(seed))
