#!/usr/bin/env python3
"""Action-leverage preflight (system1_motion_spec.md §3 Aux 1).

THE single most important check before substrate training: confirm the env's
transitions are action-dependent enough that action-conditioned prediction can't
degenerate to copy-forward. Precommit: r_action > 0.15, where

    r_action = (MSE_history_only - MSE_action_conditioned) / MSE_history_only

measured by training two small one-step predictors on collected transitions:
  history-only : MLP(state_t)            -> state_{t+1}
  action-cond  : MLP(state_t, action_t)  -> state_{t+1}

Proxy note: pre-substrate, leverage is measured on the env's TRUE state (the
Markov obs, which includes velocity, so it already encodes "history"). Image-
latent leverage is implied (state changes => rendered frame changes) and can be
re-checked once a substrate exists. r_action is a scale-invariant ratio.

    python -m gates.action_leverage_preflight --env reacher --task easy
"""
from __future__ import annotations

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from dm_control import suite


def collect(env_name, task, n_transitions, episode_len, seed):
    env = suite.load(env_name, task, task_kwargs={"random": seed})
    aspec = env.action_spec()
    rng = np.random.RandomState(seed)
    S, A, Snext = [], [], []
    ts = env.reset(); steps = 0
    def obs_vec(o): return np.concatenate([np.asarray(v).ravel() for v in o.values()]).astype(np.float32)
    s = obs_vec(ts.observation)
    while len(S) < n_transitions:
        a = rng.uniform(aspec.minimum, aspec.maximum, aspec.shape).astype(np.float32)
        ts = env.step(a); steps += 1
        snext = obs_vec(ts.observation)
        S.append(s); A.append(a); Snext.append(snext)
        s = snext
        if ts.last() or steps >= episode_len:
            ts = env.reset(); s = obs_vec(ts.observation); steps = 0
    return np.array(S), np.array(A), np.array(Snext)


def train_predictor(X, Y, steps, dev, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor(X, device=dev); Y = torch.tensor(Y, device=dev)
    n = len(X); rng = np.random.RandomState(seed); idx = rng.permutation(n); ntr = int(0.8 * n)
    tr, te = torch.tensor(idx[:ntr], device=dev), torch.tensor(idx[ntr:], device=dev)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        sel = tr[torch.randint(0, len(tr), (256,), generator=g, device=dev)]
        loss = ((net(X[sel]) - Y[sel]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((net(X[te]) - Y[te]) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="reacher")
    ap.add_argument("--task", default="easy")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--episode-len", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--out", default="gates/action_leverage_reacher.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S, A, Snext = collect(args.env, args.task, args.n, args.episode_len, args.seed)
    # z-score the prediction target (per-dim) so MSE is dimensionless; ratio is scale-invariant
    mu, sd = Snext.mean(0), Snext.std(0) + 1e-6
    Yn = (Snext - mu) / sd
    Sn = (S - mu) / sd
    copy_mse = float(((Yn - Sn) ** 2).mean())                      # trivial baseline
    mse_hist = train_predictor(Sn, Yn, args.steps, dev, args.seed)
    mse_act = train_predictor(np.concatenate([Sn, A], 1), Yn, args.steps, dev, args.seed)
    r_action = (mse_hist - mse_act) / max(mse_hist, 1e-9)
    res = {"env": f"{args.env}/{args.task}", "n_transitions": len(S),
           "copy_forward_mse": round(copy_mse, 4),
           "mse_history_only": round(mse_hist, 4),
           "mse_action_conditioned": round(mse_act, 4),
           "r_action": round(float(r_action), 4),
           "threshold": args.threshold,
           "preflight_pass": bool(r_action > args.threshold)}
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nPREFLIGHT: {'PASS' if res['preflight_pass'] else 'FAIL'} "
          f"(r_action={res['r_action']} vs {args.threshold}) on {res['env']}")


if __name__ == "__main__":
    main()
