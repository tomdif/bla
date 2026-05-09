"""Generic save/load for any nn.Module + optional optimizer + config.

The save format is a single .pt file:
    {
        "version": 1,
        "model_class": "system1_jepa.model.BLAJEPAModel",
        "config": {...},                  # dataclass.asdict if available
        "state_dict": {...},
        "optimizer_state": {...},         # optional
        "step": int,
    }

`load_into(model, path)` reads back into an existing instance — useful when
the model class is constructed elsewhere (e.g. with a Config object the
caller already has). `load_with_class(path)` reconstructs by importing the
class string. The latter trades coupling for convenience.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Optional

import torch
from torch import nn


def _qualified_name(obj: Any) -> str:
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(name: str):
    module_name, _, qual = name.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, qual)


def save(
    model: nn.Module,
    path: str,
    config: Any = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
) -> None:
    cfg_dict = None
    if config is not None:
        if dataclasses.is_dataclass(config):
            cfg_dict = dataclasses.asdict(config)
        elif isinstance(config, dict):
            cfg_dict = config

    payload = {
        "version": 1,
        "model_class": _qualified_name(model),
        "config": cfg_dict,
        "state_dict": model.state_dict(),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_into(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {"step": payload.get("step", 0), "config": payload.get("config")}


def load_with_class(path: str, **construct_kwargs) -> tuple[nn.Module, dict]:
    """Reconstruct the model from the saved class name + config.

    Pass extra construction kwargs (e.g. RAM module) via construct_kwargs;
    they are merged with the saved config dict at construction time.
    """

    payload = torch.load(path, map_location="cpu")
    cls = _import_class(payload["model_class"])
    cfg = payload.get("config")
    if cfg is not None:
        config_cls_name = payload.get("config_class")
        if config_cls_name is not None:
            config_cls = _import_class(config_cls_name)
            cfg_obj = config_cls(**cfg)
            model = cls(cfg_obj, **construct_kwargs)
        else:
            model = cls(**cfg, **construct_kwargs)
    else:
        model = cls(**construct_kwargs)
    model.load_state_dict(payload["state_dict"])
    return model, {"step": payload.get("step", 0), "config": cfg}
