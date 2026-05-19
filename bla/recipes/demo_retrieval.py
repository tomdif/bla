"""Demo retrieval — scale Recipe E from fixed cycling to NN retrieval.

The doctrine (Phase D3+D4+Scale-1): for contact-sensitive demo-prior
regimes, the demonstration manifold IS the policy. Recipe E
("demo_no_cem") replays demo actions without CEM.

This module scales Recipe E by retrieving the closest useful demo
from a bank instead of cycling through a fixed subset. The retrieval
key is the current (object, end-effector) pose; the proposal is the
matched demo's action sequence.

Phase DR1 protocol: build_index() from a list of DemoState records,
retrieve() top-k by L2 in the key space, propose() the top-1
(or averaged-top-k) action sequence.

No CEM. No learned scorer. Pure logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DemoState:
    """One record in the retrieval bank.

    key (np.ndarray, shape [D]):
        retrieval feature vector — typically (cubeA_xy, eef_xy, cubeA_z, eef_z)
        for PickPlaceCan-class tasks. Compared by L2.

    action_seq (np.ndarray, shape [T, action_dim]):
        the demo's action sequence to replay.

    init_state (np.ndarray, optional, shape [S]):
        mujoco flattened state if the env supports state-matched reset.
        None for fresh-reset deployments.

    demo_id (int): for debugging / tracking which demo was retrieved.

    metadata (dict): task-specific extras (e.g. recorded outcome).
    """
    key: np.ndarray
    action_seq: np.ndarray
    init_state: np.ndarray | None = None
    demo_id: int = -1
    metadata: dict = field(default_factory=dict)


class DemoRetriever:
    """NN retrieval over a bank of DemoState records.

    Stateless after `build_index` — `retrieve` and `propose` are
    pure functions of the query state and the indexed bank.

    The retriever does NOT score demo *quality* — it scores demo
    *similarity* to the query state. Quality filtering (e.g. drop
    demos that fail to lift the cube) is the caller's responsibility
    when building the bank.
    """

    def __init__(self) -> None:
        self._bank: list[DemoState] = []
        self._keys: np.ndarray | None = None  # [N, D]

    def build_index(self, demos: Sequence[DemoState]) -> None:
        """Store the bank and stack key vectors for batched L2 distance."""
        if len(demos) == 0:
            raise ValueError("Empty demo bank")
        keys = np.stack([d.key for d in demos], axis=0).astype(np.float32)
        if keys.ndim != 2:
            raise ValueError(f"Expected 2-D key array, got shape {keys.shape}")
        self._bank = list(demos)
        self._keys = keys

    def __len__(self) -> int:
        return len(self._bank)

    def retrieve(self, query_key: np.ndarray, k: int = 1) -> list[DemoState]:
        """Return the k bank entries closest to query_key by L2 in key space."""
        if self._keys is None:
            raise RuntimeError("build_index has not been called")
        q = np.asarray(query_key, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._keys.shape[1]:
            raise ValueError(
                f"query_key dim {q.shape[0]} != bank key dim {self._keys.shape[1]}")
        d = np.linalg.norm(self._keys - q[None, :], axis=1)
        order = np.argsort(d)[: min(k, len(self._bank))]
        return [self._bank[i] for i in order]

    def propose(self, query_key: np.ndarray, k: int = 1,
                  reduce: str = "top1", H: int | None = None) -> np.ndarray:
        """Return an action sequence proposed for query_key.

        reduce:
          "top1"  — top-1 retrieved demo's action_seq (truncated to H if given)
          "topk_avg" — elementwise mean of top-k action sequences. Note:
                    averaging blurs grasp/gripper timing; mostly useful
                    as a diagnostic mode, not a deployment default.
        """
        demos = self.retrieve(query_key, k=k)
        if reduce == "top1":
            actions = demos[0].action_seq
        elif reduce == "topk_avg":
            T_min = min(d.action_seq.shape[0] for d in demos)
            stacked = np.stack(
                [d.action_seq[:T_min] for d in demos], axis=0)
            actions = stacked.mean(axis=0).astype(np.float32)
        else:
            raise ValueError(f"unknown reduce: {reduce}")
        if H is not None:
            if actions.shape[0] >= H:
                actions = actions[:H]
            else:
                # Pad by repeating the last action
                pad = np.repeat(actions[-1:], H - actions.shape[0], axis=0)
                actions = np.concatenate([actions, pad], axis=0)
        return actions.astype(np.float32)

    def index_keys(self) -> np.ndarray:
        """Return the [N, D] key matrix of the indexed bank (debugging)."""
        if self._keys is None:
            raise RuntimeError("build_index has not been called")
        return self._keys.copy()
