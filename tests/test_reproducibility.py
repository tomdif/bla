"""Unit tests for bla.reproducibility.set_all_seeds."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from bla.reproducibility import (
    make_torch_generator,
    set_all_seeds,
    worker_init_fn,
)


def _snapshot(n: int = 4):
    """Sample one value from each RNG; used to compare runs."""
    return {
        "random": [random.random() for _ in range(n)],
        "numpy": np.random.rand(n).tolist(),
        "torch": torch.rand(n).tolist(),
    }


def test_same_seed_same_samples():
    set_all_seeds(42)
    a = _snapshot()
    set_all_seeds(42)
    b = _snapshot()
    assert a == b


def test_different_seed_different_samples():
    set_all_seeds(42)
    a = _snapshot()
    set_all_seeds(43)
    b = _snapshot()
    assert a["random"] != b["random"]
    assert a["numpy"] != b["numpy"]
    assert a["torch"] != b["torch"]


def test_returns_seed():
    assert set_all_seeds(7) == 7


def test_rejects_non_int():
    with pytest.raises(TypeError):
        set_all_seeds("42")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        set_all_seeds(42.0)  # type: ignore[arg-type]


def test_pythonhashseed_set():
    set_all_seeds(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_pythonhashseed_optout():
    os.environ.pop("PYTHONHASHSEED", None)
    set_all_seeds(99, pythonhashseed=False)
    assert "PYTHONHASHSEED" not in os.environ


def test_make_torch_generator_deterministic():
    g1 = make_torch_generator(5)
    g2 = make_torch_generator(5)
    a = torch.rand(4, generator=g1).tolist()
    b = torch.rand(4, generator=g2).tolist()
    assert a == b


def test_make_torch_generator_distinct_seeds():
    g1 = make_torch_generator(5)
    g2 = make_torch_generator(6)
    a = torch.rand(4, generator=g1).tolist()
    b = torch.rand(4, generator=g2).tolist()
    assert a != b


def test_worker_init_fn_runs_without_error():
    torch.manual_seed(0)
    worker_init_fn(0)
    worker_init_fn(3)


def test_deterministic_cudnn_flag_safe_on_cpu():
    set_all_seeds(1, deterministic_cudnn=True)
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
