"""OF-JEPA — Object-File JEPA, canonical BLA System-1 perception substrate.

Subpackage layout:

  object_file_memory.py — OFJEPAConfig, ObjectFileMemory (canonical v0),
                          ObjectFileMemoryV1 (Phase 9B falsified, kept
                          for reproducibility)
  assignment.py         — differentiable Sinkhorn for memory↔proposal binding
  encoder.py            — ConvNeXt-T proposal encoder
  predictor.py          — OFJEPA top-level wrapper (assembles encoder + memory)
  interfaces.py         — OFJEPAObjectFiles substrate API + ObjectFileBatch
  metrics.py            — identity-conditioned probe + diagnostics

Canonical import surface (preserves backward compatibility with the
pre-refactor `from system1_jepa.of_jepa import ...` calls):
"""
from .assignment import sinkhorn
from .encoder import ProposalEncoder
from .interfaces import (
    ObjectFileBatch,
    OFJEPAObjectFiles,
    per_file_project,
)
from .object_file_memory import (
    OFJEPAConfig,
    ObjectFileMemory,
    ObjectFileMemoryV1,
)
from .predictor import OFJEPA

__all__ = [
    "OFJEPA",
    "OFJEPAConfig",
    "ObjectFileMemory",
    "ObjectFileMemoryV1",
    "ObjectFileBatch",
    "OFJEPAObjectFiles",
    "ProposalEncoder",
    "per_file_project",
    "sinkhorn",
]
