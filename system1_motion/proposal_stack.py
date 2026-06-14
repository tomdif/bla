#!/usr/bin/env python3
"""Proposal-stack planner -- the architecture beyond CEM-as-core.

    proposers (policy / library / language / learned-Δ / CEM / MPPI)
        -> world-model rollout
        -> Δachievable + OOD verification
        -> governor {TRUSTED, REFINE, MEASURE, REJECT, OOD_REFUSE}
        -> release ONE action (MPC), refine locally only when useful

CEM is demoted to ONE proposer / local refiner, never the thing that directly releases actions. The stack is
world-model-agnostic: a WorldModelAdapter exposes a batched `rollout(z0, action_seqs) -> decoded_states`, so the
same stack drives the toy model below, the 2D Reacher WM, or the 3D torque WM (TorchWMAdapter). The action search
runs in numpy; only the adapter touches torch -- so the orchestration is unit-testable offline (see __main__).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, List
import numpy as np


# ----------------------------- typed hypothesis -----------------------------
@dataclass
class ActionChunkHypothesis:
    source: str                                    # "policy"|"mppi"|"cem"|"library"|"language"|"learned_delta"
    actions: np.ndarray                            # [H, adim]
    predicted_value: float                         # higher = better (e.g. -cost); planner's own score
    predicted_states: Optional[np.ndarray] = None  # [H, state_dim] decoded model rollout
    predicted_delta_achievable: Optional[float] = None   # verifier-estimated progress toward the objective
    uncertainty: float = 0.0                       # epistemic [0,1]; high -> don't trust raw
    ood: float = 0.0                               # out-of-distribution score [0,1]
    requires_verification: bool = True
    meta: dict = field(default_factory=dict)


# ----------------------------- world-model interface -----------------------------
class WorldModelAdapter(Protocol):
    adim: int
    action_low: np.ndarray
    action_high: np.ndarray
    def current_state(self, z0) -> np.ndarray: ...                  # decoded state at z0  [state_dim]
    def rollout(self, z0, action_seqs: np.ndarray) -> np.ndarray: ...  # [N,H,adim] -> [N,H,state_dim] decoded


# objective: decoded states [N,H,state_dim] -> cost [N] (lower is better)
Objective = Callable[[np.ndarray], np.ndarray]

def goal_objective(goal, terminal_w: float = 4.0) -> Objective:
    g = np.asarray(goal, np.float64)
    def obj(states):
        d = np.linalg.norm(states - g[None, None], axis=-1)        # [N,H]
        w = np.ones(states.shape[1]); w[-1] = terminal_w
        return (d * w).sum(-1)
    return obj


# ----------------------------- proposers -----------------------------
class Proposer(Protocol):
    name: str
    def propose(self, wm: WorldModelAdapter, z0, objective: Objective, horizon: int,
                rng: np.random.RandomState, warm: Optional[np.ndarray] = None) -> List[ActionChunkHypothesis]: ...

@dataclass
class CEMProposer:
    pop: int = 256; elite: int = 32; iters: int = 4; sigma0: float = 0.5; name: str = "cem"
    def propose(self, wm, z0, objective, horizon, rng, warm=None):
        lo, hi = wm.action_low, wm.action_high
        mu = warm.copy() if warm is not None else np.zeros((horizon, wm.adim))
        sigma = np.full((horizon, wm.adim), self.sigma0)
        best_seq, best_cost, best_states = None, np.inf, None
        for _ in range(self.iters):
            seqs = np.clip(mu[None] + sigma[None] * rng.randn(self.pop, horizon, wm.adim), lo, hi)
            states = wm.rollout(z0, seqs); cost = objective(states)
            idx = cost.argsort()[:self.elite]; e = seqs[idx]; mu = e.mean(0); sigma = e.std(0) + 1e-3
            if cost[idx[0]] < best_cost: best_cost, best_seq, best_states = cost[idx[0]], seqs[idx[0]], states[idx[0]]
        return [ActionChunkHypothesis("cem", best_seq, -float(best_cost), best_states,
                                      uncertainty=float(np.clip(sigma.mean() / max(1e-6, self.sigma0), 0, 1)))]

@dataclass
class MPPIProposer:
    """soft exponential weighting over ALL sampled trajectories (no hard elite truncation) -- typically
    smoother and less mode-collapsing than CEM in continuous control."""
    pop: int = 256; iters: int = 4; sigma0: float = 0.5; temperature: float = 1.0; name: str = "mppi"
    def propose(self, wm, z0, objective, horizon, rng, warm=None):
        lo, hi = wm.action_low, wm.action_high
        mu = warm.copy() if warm is not None else np.zeros((horizon, wm.adim)); w = np.ones(self.pop) / self.pop
        for _ in range(self.iters):
            seqs = np.clip(mu[None] + self.sigma0 * rng.randn(self.pop, horizon, wm.adim), lo, hi)
            cost = objective(wm.rollout(z0, seqs))
            w = np.exp(-(cost - cost.min()) / max(1e-6, self.temperature)); w = w / w.sum()
            mu = (w[:, None, None] * seqs).sum(0)
        states = wm.rollout(z0, mu[None])[0]; cost_mu = float(objective(states[None])[0])
        ess = 1.0 / float((w ** 2).sum())                          # effective sample size -> concentration = uncertainty
        return [ActionChunkHypothesis("mppi", mu, -cost_mu, states, uncertainty=float(np.clip(1 - ess / self.pop, 0, 1)))]

@dataclass
class PolicyProposer:
    """amortized policy trained in imagination (Dreamer-style). `policy(z0, horizon) -> [H,adim]`. Until trained,
    `policy=None` and this proposer simply yields nothing -- the stack falls back to CEM/MPPI."""
    policy: Optional[Callable] = None; name: str = "policy"
    def propose(self, wm, z0, objective, horizon, rng, warm=None):
        if self.policy is None: return []
        actions = np.asarray(self.policy(z0, horizon)); states = wm.rollout(z0, actions[None])[0]
        return [ActionChunkHypothesis("policy", actions, -float(objective(states[None])[0]), states, uncertainty=0.15)]


# ----------------------------- verification + governor -----------------------------
@dataclass
class DeltaAchievableVerifier:
    """estimate predicted Δachievable (progress toward objective) + an OOD score. `delta_estimator` and
    `ood_scorer` are where a LEARNED Δ-estimator / conformal-OOD gate plug in; both fall back to model-based proxies."""
    delta_estimator: Optional[Callable] = None
    ood_scorer: Optional[Callable] = None
    def verify(self, wm, z0, hyp: ActionChunkHypothesis, objective: Objective):
        if self.delta_estimator is not None:
            hyp.predicted_delta_achievable = float(self.delta_estimator(z0, hyp.actions))
        elif hyp.predicted_states is not None:                     # proxy: model-predicted progress vs staying put
            cur = wm.current_state(z0)
            c0 = float(objective(np.repeat(cur[None, None], hyp.predicted_states.shape[0], 1))[0])
            cbest = float(objective(hyp.predicted_states[None]).min())
            hyp.predicted_delta_achievable = c0 - cbest
        hyp.ood = float(self.ood_scorer(z0, hyp.actions)) if self.ood_scorer is not None \
            else float(np.mean(np.abs(hyp.actions) >= 0.98))       # proxy: fraction of saturated action dims
        return hyp

class Decision(Enum):
    TRUSTED = "trusted"; REFINE = "refine"; MEASURE = "measure"; REJECT = "reject"; OOD_REFUSE = "ood_refuse"

@dataclass
class Governor:
    min_delta: float = 0.02; max_uncertainty: float = 0.5; max_ood: float = 0.8
    def decide(self, hyp: ActionChunkHypothesis) -> Decision:
        if hyp.ood > self.max_ood: return Decision.OOD_REFUSE
        if hyp.predicted_delta_achievable is not None and hyp.predicted_delta_achievable < self.min_delta:
            return Decision.REJECT
        if hyp.uncertainty > self.max_uncertainty: return Decision.REFINE
        return Decision.TRUSTED


# ----------------------------- the stack -----------------------------
@dataclass
class PlanningStack:
    proposers: List[Proposer]
    verifier: DeltaAchievableVerifier
    governor: Governor
    refiner: Optional[Proposer] = None                            # local refinement (MPPI/CEM) on REFINE
    def plan(self, wm, z0, objective, horizon, rng=None):
        rng = rng or np.random.RandomState(0)
        hyps: List[ActionChunkHypothesis] = []
        for p in self.proposers: hyps += p.propose(wm, z0, objective, horizon, rng)
        for h in hyps: self.verifier.verify(wm, z0, h, objective)
        hyps.sort(key=lambda h: -(h.predicted_delta_achievable if h.predicted_delta_achievable is not None else h.predicted_value))
        for h in hyps:
            d = self.governor.decide(h); h.meta["decision"] = d.value
            if d is Decision.TRUSTED:
                return self._release(h, d, hyps)
            if d is Decision.REFINE and self.refiner is not None:
                hr = self.refiner.propose(wm, z0, objective, horizon, rng, warm=h.actions)
                if hr:
                    self.verifier.verify(wm, z0, hr[0], objective); hr[0].meta["decision"] = "refine"
                    if self.governor.decide(hr[0]) is not Decision.OOD_REFUSE:
                        return self._release(hr[0], Decision.REFINE, hyps)
            # REJECT / MEASURE / OOD_REFUSE -> try the next-best hypothesis
        best = hyps[0] if hyps else None                          # nothing trusted -> flagged fallback
        return {"action": (best.actions[0] if best is not None else np.zeros(wm.adim)),
                "decision": "fallback", "hypothesis": best, "all": hyps}
    def _release(self, h, d, hyps):
        return {"action": h.actions[0], "decision": d.value, "hypothesis": h, "all": hyps}


def default_stack(refine=True):
    """policy(stub) + MPPI + CEM proposers, model-based verifier, governor, MPPI local refiner."""
    return PlanningStack(proposers=[PolicyProposer(), MPPIProposer(), CEMProposer()],
                         verifier=DeltaAchievableVerifier(), governor=Governor(),
                         refiner=MPPIProposer(iters=2, pop=128) if refine else None)


# ----------------------------- torch adapter (plugs into r3 / r3_torque world models) -----------------------------
class TorchWMAdapter:
    """wraps an r3-style wm dict {dyn, dec_g, adim} so the stack can plan a real latent world model.
    z0 is a torch latent [1, d_z]; rollout decodes dec_g (normalized state) -- a drop-in for cem_plan3d."""
    def __init__(self, wm, device, action_low=-1.0, action_high=1.0):
        import torch; self.t = torch; self.wm = wm; self.dev = device; self.adim = wm["adim"]
        self.action_low = np.full(self.adim, action_low); self.action_high = np.full(self.adim, action_high)
    def current_state(self, z0):
        with self.t.no_grad(): return self.wm["dec_g"](z0)[0].cpu().numpy()
    def rollout(self, z0, action_seqs):
        N, H, _ = action_seqs.shape
        with self.t.no_grad():
            a = self.t.tensor(action_seqs, dtype=self.t.float32, device=self.dev)
            z = z0.expand(N, -1).clone(); out = []
            for h in range(H):
                z = self.wm["dyn"](z, a[:, h]); out.append(self.wm["dec_g"](z))
            return self.t.stack(out, 1).cpu().numpy()             # [N,H,state_dim]


# ----------------------------- offline self-test (numpy toy WM, no torch/GPU) -----------------------------
class _ToyAdapter:
    """trivial controllable WM: state = 3D position, each step pos += 0.15*action (clipped). Lets us verify the
    stack's orchestration (propose -> verify -> govern -> MPC) end-to-end with no dependencies."""
    adim = 3; action_low = np.full(3, -1.0); action_high = np.full(3, 1.0)
    def __init__(self, pos): self.pos = np.asarray(pos, np.float64)
    def current_state(self, z0): return z0.copy()
    def rollout(self, z0, action_seqs):
        N, H, _ = action_seqs.shape; z = np.repeat(z0[None], N, 0); out = np.empty((N, H, 3))
        for h in range(H):
            z = z + 0.15 * np.clip(action_seqs[:, h], -1, 1); out[:, h] = z
        return out


if __name__ == "__main__":
    rng = np.random.RandomState(0); goal = np.array([0.9, -0.6, 0.4]); obj = goal_objective(goal)
    stack = default_stack(); pos = np.array([-0.5, 0.5, -0.3]); wm = _ToyAdapter(pos)
    print(f"toy MPC toward {goal}  start {pos}")
    decisions = {}
    for step in range(40):
        out = stack.plan(wm, pos, obj, horizon=10, rng=rng)
        decisions[out["decision"]] = decisions.get(out["decision"], 0) + 1
        pos = pos + 0.15 * np.clip(out["action"], -1, 1)         # apply first action (MPC)
        if step % 8 == 0 or step == 39:
            h = out["hypothesis"]
            print(f"  step {step:2d} dist={np.linalg.norm(pos-goal):.3f} src={h.source if h else '-':6} "
                  f"decision={out['decision']:8} Δ={h.predicted_delta_achievable:.3f} unc={h.uncertainty:.2f}")
    print(f"  final dist={np.linalg.norm(pos-goal):.3f}  decisions={decisions}")
    assert np.linalg.norm(pos - goal) < 0.05, "stack failed to drive the toy WM to the goal"
    print("  SELF-TEST PASS: proposal stack drives a controllable WM to the goal via propose->verify->govern->MPC")
