#!/usr/bin/env python3
"""Grounding-DISSOCIATION trainer — does a grounding channel selectively encode
the variable it is indexed to, and provably NOT the others?

Q8 (the screw): grounding channels are indexed by an axis.
  - inverse-dynamics / prior grounding is indexed by CONTROLLABILITY: it forces the
    latent to encode what the action moves (the arm) — because you can't recover the
    action otherwise. It gives NO pressure to encode an uncontrolled variable.
  - a target/affordance head is indexed by DECISION-RELEVANCE: it forces the latent
    to encode the (uncontrolled) target, because that's what it's supervised on.

Reacher gives both variables for free: FINGER position (controllable) and TARGET
position (uncontrolled, set at episode reset, decision-relevant). We run three
conditions and probe BOTH variables under each:

  condition           inv  prior  tgt | predicted: arm-probe   target-probe
  C0 baseline          0    0      0  |            FAIL         FAIL
  C1 +action-ground    1   0.5     0  |            PASS         FAIL   <- dissociation
  C2 +target-ground    0    0      1  |            (fail/part)  PASS   <- control

The headline is the INTERACTION in C1: action grounding rescues the controllable
variable and provably NOT the uncontrolled one. C2 is the control proving the
target IS linearly decodable when the right channel grounds it (rules out
"target just isn't in the pixels / isn't learnable").

Everything else (encoder, dynamics, EMA target, var-hinge, disjoint clips) is held
identical to train_v4.py. Only the grounding-term weights and the two probes differ.

    python -m system1_motion.train_dissoc --data runs/reacher_transitions.npz \
        --condition C1 --inv-weight 1.0 --prior-weight 0.5 --tgt-weight 0.0
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead, ema_update
from system1_motion.slot_encoder import SlotEncoder
from system1_motion.objective import (RunningSigma, RunningStandardizer, substrate_loss,
                                       inverse_dynamics_loss, prior_grounding_loss)
from system1_motion.aux_networks import ActionEmbedder, InverseDynamics, Prior
from system1_motion.data import TransitionDataset
from system1_motion.objective import variance_hinge
from system1_motion.regularizers import sigreg, QLIFFloor, sigreg_lewm, sigreg_epps_pulley
from gates.gate0_precommit import decode_gate


def ema_momentum(step, total, m0=0.99, m1=0.9999):
    return m0 + (m1 - m0) * min(step / max(total, 1), 1.0)


def beta_ramp(step, warmup=2000, beta=1.0):
    return beta * min(step / max(warmup, 1), 1.0)


def build_eval(ds, frac=0.2):
    """Held-out (past, future, action-window-flat, finger-pos, target-pos)."""
    n = len(ds.pairs); k = int((1 - frac) * n); ends = ds.pairs[k:]; S = ds.S
    H = ds.frames.shape[2]
    past = np.stack([ds.frames[e - S + 1:e + 1].reshape(-1, H, H) for e in ends]).astype(np.uint8)
    fut = np.stack([ds.frames[e + 1:e + 1 + S].reshape(-1, H, H) for e in ends]).astype(np.uint8)
    aw = np.stack([ds.actions[e:e + S].reshape(-1) for e in ends]).astype(np.float32)
    return past, fut, aw, ds.pos[ends], ds.target[ends]


@torch.no_grad()
def _embed(enc, frames, device, bs=512):
    enc.eval(); Z = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i + bs]).float().to(device) / 255.0
        Z.append(enc(x).cpu().numpy())
    enc.train(); return np.concatenate(Z)


def probe(enc, past, gt, img_px, device, threshold_px=5.0):
    """Linear-readout (fresh decoder) probe of `gt` (px) from the latent."""
    Z = _embed(enc, past, device)
    return decode_gate(Z, gt, img_px, steps=5000, threshold_px=threshold_px, device=device)


def action_recoverability(enc, past, fut, aw_std, device, steps=3000):
    """Leading indicator: can a FRESH probe recover the action from (h_t, z_future)?
    1 - mse/var on held-out (~R^2). Rises if the encoder is action-informative —
    the thing a bottlenecked q_psi should FORCE, separate from arm-decode."""
    import torch.nn as nn
    Hp = _embed(enc, past, device); Hf = _embed(enc, fut, device)
    X = np.concatenate([Hp, Hf], 1); Y = aw_std
    Xt = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6), dtype=torch.float32, device=device)
    Yt = torch.tensor((Y - Y.mean(0)) / (Y.std(0) + 1e-6), dtype=torch.float32, device=device)
    n = len(Xt); idx = np.random.RandomState(0).permutation(n)
    tr = torch.tensor(idx[:int(.8*n)], device=device); te = torch.tensor(idx[int(.8*n):], device=device)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 256), nn.GELU(), nn.Linear(256, Yt.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3); g = torch.Generator(device=device).manual_seed(0)
    for _ in range(steps):
        s = tr[torch.randint(0, len(tr), (256,), generator=g, device=device)]
        l = ((net(Xt[s]) - Yt[s]) ** 2).mean(); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        mse = float(((net(Xt[te]) - Yt[te]) ** 2).mean())
    return round(1.0 - mse, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--condition", default="C?", help="label for outputs (C0/C1/C2)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--d-z", type=int, default=384)
    ap.add_argument("--d-u", type=int, default=32)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--dyn-depth", type=int, default=4)
    ap.add_argument("--encoder", default="pool", choices=["pool", "slot"])  # mean-pool ViT vs OF-JEPA slots
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--slot-dim", type=int, default=64)
    ap.add_argument("--vit-dim", type=int, default=192)      # slot ViT backbone width
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-var", type=float, default=1.0)      # weight on the anti-collapse term
    ap.add_argument("--reg", default="varhinge",
                    choices=["varhinge", "sigreg", "qlif", "sigreg_lewm", "sigreg_ep", "hinge_qlif"])
    ap.add_argument("--sigma-mode", default="ema", choices=["ema", "batch"])  # check-1: 'batch' = no EMA lag
    ap.add_argument("--inv-weight", type=float, default=0.0)      # mu   (action grounding)
    ap.add_argument("--prior-weight", type=float, default=0.0)    # nu   (action grounding)
    ap.add_argument("--tgt-weight", type=float, default=0.0)      # lambda (target grounding)
    ap.add_argument("--qpsi-hidden", type=int, default=256)       # inverse-dynamics head capacity
    ap.add_argument("--qpsi-linear", action="store_true")         # linear q_psi (tightest bottleneck)
    ap.add_argument("--gate-every", type=int, default=5000)
    ap.add_argument("--threshold-px", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/system1_motion_dissoc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or args.device in ("cpu", "mps")) else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)
    print(f"[dissoc] condition={args.condition} inv={args.inv_weight} prior={args.prior_weight} "
          f"tgt={args.tgt_weight} seed={args.seed} dev={dev}", flush=True)

    ds = TransitionDataset(args.data, frame_stack=args.frame_stack, disjoint=True, return_target=True)
    img_px = ds.img_px; da = ds.actions.shape[1]; aw_dim = args.frame_stack * da
    e_past, e_fut, e_aw, e_pos, e_tgt = build_eval(ds, 0.2)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    in_ch = 3 * args.frame_stack

    if args.encoder == "slot":
        enc = SlotEncoder(in_channels=in_ch, vit_dim=args.vit_dim, patch=args.patch,
                          depth=args.enc_depth, n_slots=args.n_slots, slot_dim=args.slot_dim).to(dev)
        args.d_z = enc.d_z                                   # override: latent = n_slots*slot_dim
        print(f"[dissoc] OF-JEPA slot encoder: {args.n_slots} slots x {args.slot_dim} = d_z {args.d_z}", flush=True)
    else:
        enc = ViTEncoder(args.image_size, args.patch, in_ch, args.d_z, args.enc_depth).to(dev)
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    dyn = LatentDynamics(args.d_z, args.d_u, args.dyn_depth).to(dev)
    E_a = ActionEmbedder(aw_dim, args.d_u).to(dev)
    q_psi = InverseDynamics(args.d_z, args.d_u, hidden=args.qpsi_hidden, linear=args.qpsi_linear).to(dev)
    p_rho = Prior(args.d_z, args.d_u).to(dev)
    # target-grounding head — trained INTO the encoder (NOT detached). This is the
    # decision-relevance channel: it forces the latent to encode the target.
    tgt_ground = DecodeHead(args.d_z, out_dim=e_tgt.shape[1]).to(dev)
    # detached diagnostic decoders (never enter substrate loss) for live logging
    dec_arm = DecodeHead(args.d_z, out_dim=e_pos.shape[1]).to(dev)

    qlif = QLIFFloor(args.d_z, args.d_u).to(dev) if args.reg == "qlif" else None
    print(f"[dissoc] anti-collapse reg = {args.reg}", flush=True)
    sub_params = (list(enc.parameters()) + list(dyn.parameters()) + list(E_a.parameters())
                  + list(q_psi.parameters()) + list(p_rho.parameters()) + list(tgt_ground.parameters())
                  + (list(qlif.parameters()) if qlif is not None else []))
    opt_sub = torch.optim.AdamW(sub_params, lr=args.lr, weight_decay=1e-4)
    opt_dec = torch.optim.AdamW(dec_arm.parameters(), lr=1e-3, weight_decay=1e-4)
    sigma = RunningSigma(); astd = RunningStandardizer()

    def both_probes(step):
        g_arm = probe(enc, e_past, e_pos, img_px, dev, args.threshold_px)
        g_tg = probe(enc, e_past, e_tgt, img_px, dev, args.threshold_px)
        aw_std = astd(torch.tensor(e_aw, device=dev)).cpu().numpy()
        ar = action_recoverability(enc, e_past, e_fut, aw_std, dev)
        rec = {"step": step,
               "arm_px": g_arm["mean_l2_px"], "arm_pass": g_arm["pass"],
               "target_px": g_tg["mean_l2_px"], "target_pass": g_tg["pass"],
               "action_recoverability": ar}
        print(f"[PROBE @ {step}] {args.condition} | "
              f"ARM(controllable)={g_arm['mean_l2_px']}px {'PASS' if g_arm['pass'] else 'FAIL'} | "
              f"TARGET(uncontrolled)={g_tg['mean_l2_px']}px {'PASS' if g_tg['pass'] else 'FAIL'} | "
              f"action_recov={ar}", flush=True)
        return rec

    trace, t0, step = [], time.time(), 0
    di = iter(dl)
    while step < args.steps:
        try:
            past, aw, fut, pos, tg = next(di)
        except StopIteration:
            di = iter(dl); past, aw, fut, pos, tg = next(di)
        past, fut, pos, tg = past.to(dev), fut.to(dev), pos.to(dev), tg.to(dev)
        aw = aw.reshape(aw.shape[0], -1).to(dev)
        astd.update(aw); a_std = astd(aw)
        u = E_a(a_std)

        z_t = enc(past)
        with torch.no_grad():
            z_future = tgt(fut); sigma.update(z_future)
        z_pred = dyn(z_t, u)
        # sigma normalization mode (check-1: the EMA 'ema' lags the encoder scale, which
        # can ratchet collapse; 'batch' uses current-batch variance = no lag).
        if args.sigma_mode == "batch":
            sig2 = z_future.var(0, unbiased=False).detach() + 1e-4
        else:
            sig2 = sigma.sigma2(args.d_z, dev)
        # pure prediction (beta_var=0), then add the SELECTED anti-collapse regularizer
        L_pred, parts = substrate_loss(z_pred, z_future, z_t, sig2, beta_var=0.0)
        # anti-collapse FLOOR at FULL strength from step 0 (no ramp race); the
        # distinctive term ramps in. The floor is the only term that reliably
        # prevents collapse, so it must never be weak.
        hinge = variance_hinge(z_t)
        bw = beta_ramp(step, beta=args.beta_var)
        if args.reg == "sigreg_lewm":
            L_reg = bw * sigreg_lewm(z_t)                 # STANDALONE known-good SIGReg (LeWorld variant)
        elif args.reg == "sigreg_ep":
            L_reg = bw * sigreg_epps_pulley(z_t)          # STANDALONE known-good SIGReg (Epps-Pulley)
        elif args.reg == "sigreg":
            L_reg = bw * sigreg(z_t)                      # naive moment-matching (kill-tested; collapses)
        elif args.reg in ("qlif", "hinge_qlif"):
            qloss, qparts = qlif(z_t, z_future, u.detach()); parts.update(qparts)
            L_reg = args.beta_var * hinge + bw * qloss    # floor FIRST + predictive-info (collapse-guarded)
        else:
            L_reg = args.beta_var * hinge                 # varhinge: floor only
        parts["var"] = float(hinge.detach())
        L_pred = L_pred + L_reg
        L_inv = inverse_dynamics_loss(q_psi, z_t, z_future, u.detach())
        L_prior = prior_grounding_loss(p_rho, z_t, u.detach())
        # target grounding (decision-relevance channel): supervise z_t -> target px.
        L_tgt = ((tgt_ground(z_t) - tg) ** 2).mean()
        L = L_pred + args.inv_weight * L_inv + args.prior_weight * L_prior + args.tgt_weight * L_tgt
        opt_sub.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(sub_params, 1.0); opt_sub.step()
        ema_update(tgt, enc, ema_momentum(step, args.steps))

        # detached arm diagnostic
        L_dec = ((dec_arm(z_t.detach()) - pos) ** 2).mean()
        opt_dec.zero_grad(set_to_none=True); L_dec.backward(); opt_dec.step()

        step += 1
        if step % 200 == 0:
            print(f"[step {step}/{args.steps}] pred={parts['pred']:.4f} var={parts['var']:.4f} "
                  f"inv={float(L_inv):.4f} prior={float(L_prior):.4f} tgt={float(L_tgt):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if step % args.gate_every == 0 or step == args.steps:
            rec = both_probes(step)
            trace.append(rec)
            torch.save({"enc": enc.state_dict(), "args": vars(args)},
                       os.path.join(args.out_dir, f"substrate_{args.encoder}_{args.condition}_s{args.seed}.pt"))
            json.dump({"condition": args.condition, "seed": args.seed,
                       "weights": {"inv": args.inv_weight, "prior": args.prior_weight, "tgt": args.tgt_weight},
                       "trace": trace, "final": trace[-1]},
                      open(os.path.join(args.out_dir, f"trace_{args.encoder}_{args.condition}_s{args.seed}.json"), "w"), indent=2)

    print(json.dumps({"condition": args.condition, "seed": args.seed, "final": trace[-1] if trace else None}, indent=2))


if __name__ == "__main__":
    main()
