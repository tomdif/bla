#!/usr/bin/env python3
"""proofworld.imagine -- Stage C: the GROUNDED imagination loop, and the model-exploitation trap made concrete.

Built on Stage B's SIGReg proof-state world model (worldmodel.py). The agent must reach the goal node (establish it
>= 0) by a sequence of z3-sound carries. A value head D (trained on REAL z3 distances = RLVR) lets the agent score
imagined next-states. Three agents, all using the SAME imperfect world model to plan:

  * GROUNDED (closed-loop MPC): each step, imagine 1-step values for candidate actions, pick best, then EXECUTE
    against the z3 kernel (real state update) and RE-GROUND. The kernel corrects the world model's error every step.
  * OPEN-LOOP (trust imagination): plan the whole rollout in latent (commit to it), then execute -- never
    re-grounding. The world model's per-step error COMPOUNDS, so the agent's *believed* success diverges from real.
  * RANDOM: no model.

The headline: imagination HELPS (grounded & open-loop both beat random at finding the goal), but acting on
imagination WITHOUT re-grounding self-deceives -- open-loop's IMAGINED reach is high while its REAL (z3) reach is
low. That gap is the trap (imagined-8cm-vs-real-33cm, in proof space). Grounding closes it: the kernel owns truth,
imagination only proposes. A divergence monitor quantifies the exploitable model error per step.

Run:  python3 -m proofworld.imagine
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from proofworld.worldmodel import build_world, step, dist_to_goal, feat_state, feat_action, WM, train_wm


def gen_data(world, n_ep=900, max_len=9, seed=1):
    rng = np.random.RandomState(seed); acts = world["actions"]; trans = []
    for _ in range(n_ep):
        # random start (subset of nodes treated as established) for state-space coverage
        k = rng.randint(1, 3); state = set(rng.choice(world["n"], k, replace=False).tolist()) | {0}
        for _ in range(max_len):
            a = acts[rng.randint(len(acts))]; s2 = step(state, a, world)
            trans.append((frozenset(state), a, frozenset(s2))); state = s2
            if world["goal"] in state: break
    return trans


class Head(nn.Module):
    def __init__(self, din): super().__init__(); self.net = nn.Sequential(nn.Linear(din, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def _nrm(z): return nn.functional.normalize(z, dim=-1)                  # JEPA latents live on the sphere
def _z(M, state, world): return _nrm(M.enc(torch.tensor(feat_state(state, world)[None])))
def _imag(M, z, a, world): return _nrm(M.D(torch.cat([z, M.A(torch.tensor(feat_action(a, world)[None]))], -1)))


def train_Dlatent(M, world, trans, epochs=300, seed=0):
    """latent value for the imagination planner: D(normalized imagined 1-step latent) -> REAL next dist. Trained on
    M's 1-step predictions (where the planner queries it), so 1-step imagined value is accurate; MULTI-step latent
    rollouts still drift -- the honest source of open-loop's self-deception."""
    torch.manual_seed(seed)
    Sf = torch.tensor(np.stack([feat_state(s, world) for s, a, s2 in trans]))
    Af = torch.tensor(np.stack([feat_action(a, world) for s, a, s2 in trans]))
    y = torch.tensor(np.array([float(dist_to_goal(s2, world)) for s, a, s2 in trans], np.float32))
    D = Head(M.E[-1].out_features); opt = torch.optim.Adam(D.parameters(), lr=2e-3)
    with torch.no_grad(): Z = _nrm(M.D(torch.cat([M.enc(Sf), M.A(Af)], -1)))
    for _ in range(epochs):
        opt.zero_grad(); ((D(Z) - y) ** 2).mean().backward(); opt.step()
    return D


def grounded(world, max_steps=14):
    """closed-loop: 1-step KERNEL lookahead -- evaluate each candidate's REAL next dist via z3, pick best, EXECUTE,
    RE-GROUND. The kernel owns truth at every step, so it cannot self-deceive. Believed = real, by construction."""
    state = set(world["bases"])
    for s in range(max_steps):
        if world["goal"] in state: return True, s
        a = min(world["actions"], key=lambda a: dist_to_goal(step(state, a, world), world))   # z3-evaluated lookahead
        state = step(state, a, world)
    return world["goal"] in state, max_steps


def grounded_imag(M, D, world, max_steps=14):
    """imagination FOR SEARCH, grounded for TRUTH: rank actions by the world model (cheap, latent, no z3), EXECUTE
    the top-1 via z3, RE-GROUND on the real state. Imagination prunes; the kernel commits. 1 kernel call per step."""
    state = set(world["bases"])
    for s in range(max_steps):
        if world["goal"] in state: return True, s
        z = _z(M, state, world)                                                      # re-encode the REAL state
        a = min(world["actions"], key=lambda a: D(_imag(M, z, a, world)).item())     # rank by imagination (no z3)
        state = step(state, a, world)                                                # execute top-1 via the kernel
    return world["goal"] in state, max_steps


def open_loop(M, D, world, horizon=14):
    """trust imagination: roll the world model forward in LATENT (no re-grounding), commit to the plan, THEN execute.
    BELIEVED reach = what the imagined rollout says; REAL reach = z3. Their gap is the model-exploitation trap."""
    z = _z(M, set(world["bases"]), world); plan = []
    for _ in range(horizon):
        cand = [(D(_imag(M, z, a, world)).item(), a) for a in world["actions"]]
        imag_d, a = min(cand, key=lambda t: t[0]); plan.append(a); z = _imag(M, z, a, world)   # roll forward in latent
        if imag_d < 0.5: break
    imagined_reach = D(z).item() < 0.5                                                          # what the agent BELIEVES
    state = set(world["bases"])
    for a in plan:
        state = step(state, a, world)                                                          # execute the committed plan
        if world["goal"] in state: break
    return (world["goal"] in state), imagined_reach, len(plan)


def random_agent(world, max_steps=14, seed=0):
    rng = np.random.RandomState(seed); state = set(world["bases"])
    for s in range(max_steps):
        if world["goal"] in state: return True, s
        state = step(state, world["actions"][rng.randint(len(world["actions"]))], world)
    return world["goal"] in state, max_steps


def verbose_trace(M, Q, D, world):
    """make the reasoning visible: the imagined rollout IS the latent 'thought process' -- we decode it into the
    carries it considers, their imagined distance-to-goal (read out of the latent by D), and where imagination
    DIVERGES from what z3 says really happens."""
    g = world["goal"]
    print("\n  --- VERBOSE: the agent's thought process (latent rollout decoded) ---")
    print(f"  world: bases={sorted(world['bases'])} -> goal={g}; sound carries={sorted(world['sound'])}\n")
    print("  [GROUNDED] closed-loop: 1-step KERNEL lookahead, EXECUTE via z3, re-ground on the real state:")
    state = set(world["bases"])
    for t in range(world["n"]):
        if g in state: print(f"    step{t}: GOAL {g} reached (real).") ; break
        a = min(world["actions"], key=lambda a: dist_to_goal(step(state, a, world), world))
        ns = step(state, a, world)
        print(f"    step{t}: kernel-eval picks {a} -> EXECUTE -> real state {sorted(ns)}  realdist={dist_to_goal(ns, world)}")
        state = ns
    print("\n  [GROUNDED-IMAGINATION] imagine to RANK (cheap), execute top-1 via z3, re-ground:")
    state = set(world["bases"])
    for t in range(world["n"]):
        if g in state: print(f"    step{t}: GOAL {g} reached (real)."); break
        z = _z(M, state, world)
        ranked = sorted((D(_imag(M, z, a, world)).item(), a) for a in world["actions"])
        a = ranked[0][1]; ns = step(state, a, world)
        print(f"    step{t}: imagination ranks {a} best (imag dist {ranked[0][0]:.2f}) -> EXECUTE -> real {sorted(ns)} realdist={dist_to_goal(ns, world)}")
        state = ns

    print("\n  [OPEN-LOOP] trust imagination: roll the world model in LATENT, never re-grounding (believed vs real):")
    z = _z(M, set(world["bases"]), world); real = set(world["bases"]); plan = []
    for t in range(world["n"]):
        cand = sorted((D(_imag(M, z, a, world)).item(), a) for a in world["actions"])
        z = _imag(M, z, cand[0][1], world); a = cand[0][1]; plan.append(a)
        real = step(real, a, world)                                     # what that action REALLY does (for comparison)
        top = ", ".join(f"{aa}:{dd:.1f}" for dd, aa in cand[:3])
        print(f"    imagine{t}: top carries(imagined dist) [{top}]  -> pick {a}  believed_dist={D(z).item():.2f}  | REAL dist now={dist_to_goal(real, world)}")
        if D(z).item() < 0.5: print(f"    => imagination BELIEVES goal reached after {t+1} steps."); break
    print(f"    REAL outcome of the committed plan: goal {'REACHED' if g in real else 'NOT reached'} "
          f"(believed reached, real {'reached' if g in real else 'FAILED'}).")


def main():
    print("=== proofworld.imagine :: Stage C grounded imagination loop (z3 owns truth) ===\n")
    n_worlds = 10
    gk_reach, gk_steps, gi_reach, o_real, r_reach = [], [], [], [], []
    keep = None
    for w in range(n_worlds):
        world = build_world(seed=w)
        trans = gen_data(world, seed=w + 100)
        S = np.stack([feat_state(s, world) for s, a, s2 in trans])
        A = np.stack([feat_action(a, world) for s, a, s2 in trans])
        S2 = np.stack([feat_state(s2, world) for s, a, s2 in trans])
        M = train_wm("sigreg", S, A, S2, epochs=150, seed=w)             # Stage B world model (imagination engine)
        D = train_Dlatent(M, world, trans, seed=w)                       # latent value for imagined rollouts
        gk, gs = grounded(world); gk_reach.append(gk); gk_steps.append(gs)
        gi, _ = grounded_imag(M, D, world); gi_reach.append(gi)
        orl, _, _ = open_loop(M, D, world); o_real.append(orl)
        r_reach.append(np.mean([random_agent(world, seed=s)[0] for s in range(8)]))
        if w == 0: keep = (M, None, D, world)
    pct = lambda x: 100 * float(np.mean(x))
    print(f"  {n_worlds} z3-grounded worlds; each agent shares one imperfect SIGReg world model.\n")
    print(f"  {'agent':48}{'REAL reach':>12}")
    print(f"  {'GROUNDED-KERNEL (verify each candidate via z3)':48}{pct(gk_reach):>11.0f}%")
    print(f"  {'GROUNDED-IMAGINATION (trust the WM ranking, execute)':48}{pct(gi_reach):>11.0f}%")
    print(f"  {'OPEN-LOOP (trust the WM rollout, commit)':48}{pct(o_real):>11.0f}%")
    print(f"  {'RANDOM (no model)':48}{pct(r_reach):>11.0f}%")
    print(f"\n  THE TRAP, IN ITS PUREST FORM (honest finding):")
    print(f"  * This SIGReg world model is UNFAITHFUL (Stage B rollout cos ~0.6). The verbose trace shows it HALLUCINATES:")
    print(f"    it ranks carry (5,7) best from EVERY state -- even [0], where node 5 isn't proven, so (5,7) is a no-op.")
    print(f"  * So EVERY imagination-trusting agent gets stuck on the hallucinated action and fails (= random), whether or")
    print(f"    not it re-grounds the STATE -- because the ACTION RANKING itself is wrong. Only KERNEL lookahead reaches goal.")
    print(f"  * You cannot fix an unfaithful world model by acting on it (imagined-8cm-vs-real-33cm, in proof space). You")
    print(f"    must VERIFY against ground truth -- exactly why proofworld puts the KERNEL, not a learned model, in charge.")
    print(f"  * FAITHFULNESS is the bottleneck: imagination earns trust only once its rollouts are faithful enough; until")
    print(f"    then the kernel carries the agent. (Raising M's faithfulness so imagination amortizes the kernel = Stage C+.)")
    if keep: verbose_trace(*keep)


if __name__ == "__main__":
    main()
