#!/usr/bin/env python3
"""v4 substrate trainer (grounded recipe) — system1_motion_spec v4.

L_v4 = L_pred-obs + mu*L_inv + nu*L_prior + beta*L_var-hinge
(L_KL-free skipped: dataset is fully action-annotated, |D_a|:|D_f| = 1:0.)

Disjoint future-clip prediction (A3 infra). E_a embeds the standardized flattened
action window and feeds the dynamics (so E_a trains via L_pred only); sg[E_a(a)]
grounds q_psi (inverse dynamics, MSE) and p_rho (prior, Gaussian NLL). Single
optimizer; gradient flow enforced by stop-grads (verified by the gauntlet).

Tracks three diagnostics per checkpoint (Step 5): position-decode (Gate 0),
action-decode probe (is the encoder action-informative — the leading indicator),
and prior NLL on held-out. Kill-at-5k: Gate 0 >= kill_px => Case C, stop.

    python -m system1_motion.train_v4 --data runs/reacher_transitions.npz --steps 30000
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead, ema_update
from system1_motion.objective import (RunningSigma, RunningStandardizer, substrate_loss,
                                       inverse_dynamics_loss, prior_grounding_loss)
from system1_motion.aux_networks import ActionEmbedder, InverseDynamics, Prior
from system1_motion.data import TransitionDataset
from gates.gate0_precommit import decode_gate


def ema_momentum(step, total, m0=0.99, m1=0.9999):
    return m0 + (m1 - m0) * min(step / max(total, 1), 1.0)


def beta_ramp(step, warmup=2000, beta=1.0):
    return beta * min(step / max(warmup, 1), 1.0)


def build_eval(ds, frac=0.2):
    """Held-out (past, future, action-window-flat, pos) for the v4 diagnostics."""
    n = len(ds.pairs); k = int((1 - frac) * n); ends = ds.pairs[k:]; S = ds.S
    H = ds.frames.shape[2]
    past = np.stack([ds.frames[e - S + 1:e + 1].reshape(-1, H, H) for e in ends]).astype(np.uint8)
    fut = np.stack([ds.frames[e + 1:e + 1 + S].reshape(-1, H, H) for e in ends]).astype(np.uint8)
    aw = np.stack([ds.actions[e:e + S].reshape(-1) for e in ends]).astype(np.float32)
    return past, fut, aw, ds.pos[ends]


@torch.no_grad()
def _embed(enc, frames, device, bs=512):
    enc.eval(); Z = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i + bs]).float().to(device) / 255.0
        Z.append(enc(x).cpu().numpy())
    enc.train(); return np.concatenate(Z)


def gate0_check(enc, past, pos, img_px, device, threshold_px=5.0):
    Z = _embed(enc, past, device)
    return decode_gate(Z, pos, img_px, steps=5000, threshold_px=threshold_px, device=device)


def action_probe(enc, past, fut, aw_std, device, steps=3000):
    """Leading indicator: can a FRESH probe recover the action window from
    (h_t, z_future)? recoverability = 1 - mse/var_target on held-out. High => the
    encoder's features are action-informative."""
    import torch.nn as nn
    Hp = _embed(enc, past, device); Hf = _embed(enc, fut, device)
    X = np.concatenate([Hp, Hf], 1); Y = aw_std
    mu, sd = Y.mean(0), Y.std(0) + 1e-6; Yn = (Y - mu) / sd
    Xt = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6), dtype=torch.float32, device=device)
    Yt = torch.tensor(Yn, dtype=torch.float32, device=device)
    n = len(Xt); idx = np.random.RandomState(0).permutation(n); tr = torch.tensor(idx[:int(.8*n)], device=device); te = torch.tensor(idx[int(.8*n):], device=device)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.GELU(), nn.Linear(256, Yt.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3); g = torch.Generator(device=device).manual_seed(0)
    for _ in range(steps):
        s = tr[torch.randint(0, len(tr), (256,), generator=g, device=device)]
        l = ((net(Xt[s]) - Yt[s]) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        mse = float(((net(Xt[te]) - Yt[te]) ** 2).mean())          # Yn has unit var/dim
    return round(1.0 - mse, 4)                                     # recoverability ~ R^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--d-z", type=int, default=384)
    ap.add_argument("--d-u", type=int, default=32)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--dyn-depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-var", type=float, default=1.0)
    ap.add_argument("--inv-weight", type=float, default=1.0)     # mu
    ap.add_argument("--prior-weight", type=float, default=0.5)   # nu
    ap.add_argument("--gate-every", type=int, default=5000)
    ap.add_argument("--kill-step", type=int, default=5000)
    ap.add_argument("--kill-px", type=float, default=17.0)       # Case C: >= chance at 5k
    ap.add_argument("--threshold-px", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/system1_motion_v4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)

    ds = TransitionDataset(args.data, frame_stack=args.frame_stack, disjoint=True)
    img_px = ds.img_px; da = ds.actions.shape[1]; aw_dim = args.frame_stack * da
    e_past, e_fut, e_aw, e_pos = build_eval(ds, 0.2)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    in_ch = 3 * args.frame_stack

    enc = ViTEncoder(args.image_size, args.patch, in_ch, args.d_z, args.enc_depth).to(dev)
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    dyn = LatentDynamics(args.d_z, args.d_u, args.dyn_depth).to(dev)     # consumes u = E_a(a)
    E_a = ActionEmbedder(aw_dim, args.d_u).to(dev)
    q_psi = InverseDynamics(args.d_z, args.d_u).to(dev)
    p_rho = Prior(args.d_z, args.d_u).to(dev)
    dec = DecodeHead(args.d_z, out_dim=e_pos.shape[1]).to(dev)
    sub_params = (list(enc.parameters()) + list(dyn.parameters()) + list(E_a.parameters())
                  + list(q_psi.parameters()) + list(p_rho.parameters()))
    opt_sub = torch.optim.AdamW(sub_params, lr=args.lr, weight_decay=1e-4)
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    sigma = RunningSigma(); astd = RunningStandardizer()

    def astd_norm(aw_flat):
        astd.update(aw_flat); return astd(aw_flat)

    # held-out standardized action window for the action-decode probe (fixed stats after warmup)
    trace, t0, step = [], time.time(), 0
    di = iter(dl)
    while step < args.steps:
        try:
            past, aw, fut, pos = next(di)
        except StopIteration:
            di = iter(dl); past, aw, fut, pos = next(di)
        past, fut, pos = past.to(dev), fut.to(dev), pos.to(dev)
        aw = aw.reshape(aw.shape[0], -1).to(dev)          # [B, S*da]
        a_std = astd_norm(aw)
        u = E_a(a_std)                                    # live -> grad to E_a via L_pred only

        z_t = enc(past)
        with torch.no_grad():
            z_future = tgt(fut); sigma.update(z_future)
        z_pred = dyn(z_t, u)
        L_pred, parts = substrate_loss(z_pred, z_future, z_t, sigma.sigma2(args.d_z, dev),
                                       beta_var=beta_ramp(step, beta=args.beta_var))
        L_inv = inverse_dynamics_loss(q_psi, z_t, z_future, u.detach())     # sg[E_a], sg[future]
        L_prior = prior_grounding_loss(p_rho, z_t, u.detach())
        L = L_pred + args.inv_weight * L_inv + args.prior_weight * L_prior
        opt_sub.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(sub_params, 1.0); opt_sub.step()
        ema_update(tgt, enc, ema_momentum(step, args.steps))

        p_pred = dec(z_t.detach())                        # detached decode diagnostic
        L_dec = ((p_pred - pos) ** 2).mean()
        opt_dec.zero_grad(set_to_none=True); L_dec.backward(); opt_dec.step()

        step += 1
        if step % 200 == 0:
            print(f"[step {step}/{args.steps}] pred={parts['pred']:.4f} var={parts['var']:.4f} "
                  f"inv={float(L_inv):.4f} prior={float(L_prior):.4f} dec_aux={float(L_dec):.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.gate_every == 0 or step == args.steps:
            g = gate0_check(enc, e_past, e_pos, img_px, dev, args.threshold_px)
            astd_eval = astd(torch.tensor(e_aw, device=dev)).cpu().numpy()
            ar = action_probe(enc, e_past, e_fut, astd_eval, dev)
            g.update({"step": step, "action_decode_recoverability": ar, "prior_nll": float(L_prior)})
            trace.append(g)
            print(f"[GATE0 @ {step}] pos={g['mean_l2_px']}px {'PASS' if g['pass'] else 'FAIL'} | "
                  f"action_decode={ar} | prior_nll={float(L_prior):.3f}", flush=True)
            torch.save({"enc": enc.state_dict(), "dyn": dyn.state_dict(), "E_a": E_a.state_dict(),
                        "q_psi": q_psi.state_dict(), "p_rho": p_rho.state_dict(), "args": vars(args)},
                       os.path.join(args.out_dir, "substrate.pt"))
            json.dump({"trace": trace, "final": trace[-1]},
                      open(os.path.join(args.out_dir, "gate0_trace.json"), "w"), indent=2)
            if step == args.kill_step and g["mean_l2_px"] >= args.kill_px:
                print(f"[KILL] step {step}: pos {g['mean_l2_px']}px >= {args.kill_px}px (Case C). "
                      f"action_decode={ar} — stopping.", flush=True)
                break

    print(json.dumps({"final": trace[-1] if trace else None, "out_dir": args.out_dir}, indent=2))


if __name__ == "__main__":
    main()
