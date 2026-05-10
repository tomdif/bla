"""ExecutableMemory — typed registry of callable tools that B.L.A. can
invoke from the router (`RouterActionType.SIMULATE`,
`RouterActionType.PROVE`, etc.).

Tools are first-class memory entries: addressable by name, typed by
input/output schema, and uniformly invokable via `ExecutableMemory.run`.
The contract is intentionally minimal — each tool is a callable plus a
signature dict — so the same registry can hold a SymPy simplifier, a
Z3 solver, a Python sandbox, and a world-model rollout.

Phase 2 ships four built-in tools:
  * "python_sandbox"      — exec a Python source string in a subprocess
  * "sympy_simplify"      — SymPy structural equality check
  * "z3_satcheck"         — Z3 SAT/SMT discharge
  * "navigate_simulator"  — roll out a NavigateEnv N steps and report success

Real B.L.A. deployments will register dozens more (proof checkers,
unit-test runners, retrieval-augmented search, robotics simulators).
The registry stays the contract; the tools are pluggable.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class ToolEntry:
    name: str
    fn: Callable[..., Any]
    signature: dict
    description: str = ""
    metadata: dict = field(default_factory=dict)


class ExecutableMemory:
    """Typed tool registry.

    Tools are registered by name and addressable via `run(name, **kwargs)`.
    Argument schemas are advisory in Phase 2 — Phase 4+ may enforce them
    via Pydantic or similar.
    """

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self._register_builtins()

    # --- public API -----------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        signature: dict,
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} already registered")
        self._tools[name] = ToolEntry(
            name=name, fn=fn, signature=signature,
            description=description, metadata=dict(metadata or {}),
        )

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "signature": e.signature,
                "description": e.description,
            }
            for e in self._tools.values()
        ]

    def get(self, name: str) -> ToolEntry:
        if name not in self._tools:
            raise KeyError(f"tool {name!r} not in registry; have: {sorted(self._tools)}")
        return self._tools[name]

    def run(self, name: str, **kwargs) -> dict:
        """Invoke a tool. Returns {"ok": bool, "result": ..., "error": ...}."""
        try:
            entry = self.get(name)
        except KeyError as exc:
            return {"ok": False, "result": None, "error": str(exc)}
        try:
            result = entry.fn(**kwargs)
            return {"ok": True, "result": result, "error": None}
        except Exception as exc:
            return {"ok": False, "result": None, "error": f"{type(exc).__name__}: {exc}"}

    # --- built-in tools -------------------------------------------------

    def _register_builtins(self) -> None:
        self.register(
            name="python_sandbox",
            fn=_run_python_sandbox,
            signature={
                "args": {"source": "str", "timeout_s": "float?"},
                "returns": {"stdout": "str", "stderr": "str", "returncode": "int"},
            },
            description="Run a Python source string in a subprocess with a timeout.",
        )
        self.register(
            name="sympy_simplify",
            fn=_run_sympy_simplify,
            signature={
                "args": {"lhs": "str", "rhs": "str"},
                "returns": {"equal": "bool", "simplified_diff": "str"},
            },
            description="Decide whether lhs - rhs simplifies to 0.",
        )
        self.register(
            name="z3_satcheck",
            fn=_run_z3_satcheck,
            signature={
                "args": {"proposition": "str"},
                "returns": {"valid": "bool", "z3_result": "str"},
            },
            description="Discharge a proposition by negation + Z3 unsat check.",
        )
        self.register(
            name="navigate_simulator",
            fn=_run_navigate_simulator,
            signature={
                "args": {
                    "policy_state_path": "str",
                    "n_episodes": "int",
                    "n_targets": "int?",
                    "image_size": "int?",
                    "max_steps": "int?",
                    "seed": "int?",
                },
                "returns": {"success_rate": "float", "successes": "int", "total": "int"},
            },
            description="Roll out a saved BLA policy on the multi-target navigate task.",
        )


# --- tool implementations ----------------------------------------------

def _run_python_sandbox(source: str, timeout_s: float = 5.0) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "candidate.py"
        path.write_text(textwrap.dedent(source))
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=d, capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": f"timeout after {timeout_s}s", "returncode": -1}
        return {
            "stdout": proc.stdout[-4096:],
            "stderr": proc.stderr[-4096:],
            "returncode": proc.returncode,
        }


def _run_sympy_simplify(lhs: str, rhs: str) -> dict:
    import sympy
    l = sympy.sympify(lhs)
    r = sympy.sympify(rhs)
    diff = sympy.simplify(l - r)
    return {"equal": diff == 0, "simplified_diff": str(diff)}


def _run_z3_satcheck(proposition: str) -> dict:
    import z3  # type: ignore
    solver = z3.Solver()
    ctx = {**z3.__dict__}
    negated = z3.Not(eval(proposition, ctx))
    solver.add(negated)
    result = solver.check()
    return {"valid": result == z3.unsat, "z3_result": str(result)}


def _run_navigate_simulator(
    policy_state_path: str,
    n_episodes: int = 32,
    n_targets: int = 2,
    image_size: int = 16,
    max_steps: int = 14,
    seed: int = 99,
) -> dict:
    import torch
    from system1_jepa import (
        MultiTargetNavigateEnv, MultiTargetNavigateSpec, PatchViTEncoder,
    )
    from scripts.bla_multitarget import RecurrentBCPolicy

    state = torch.load(policy_state_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    encoder = PatchViTEncoder(
        in_channels=3, latent_dim=cfg["d"], patch_size=cfg["patch_size"],
        depth=cfg["encoder_depth"], heads=cfg["encoder_heads"],
    )
    policy = RecurrentBCPolicy(encoder, cfg["d"], ssm_layers=cfg["ssm_layers"])
    policy.load_state_dict(state["policy"])
    policy.eval()

    spec = MultiTargetNavigateSpec(
        image_size=image_size, patch_size=cfg["patch_size"], n_targets=n_targets,
        max_steps=max_steps, action_dim=cfg["d"],
    )
    bs = min(n_episodes, 16)
    n_batches = (n_episodes + bs - 1) // bs
    successes = 0
    total = 0
    with torch.no_grad():
        for b in range(n_batches):
            env = MultiTargetNavigateEnv(spec, batch_size=bs, seed=seed + b)
            obs = env.reset()
            history = [obs]
            for _ in range(spec.max_steps):
                obs_seq = torch.stack(history, dim=1)
                actions = policy(obs_seq)
                obs, _, done = env.step(actions[:, -1])
                history.append(obs)
                if done.all():
                    break
            successes += int(env.success_mask().sum())
            total += bs
    return {"success_rate": successes / max(total, 1), "successes": successes, "total": total}
