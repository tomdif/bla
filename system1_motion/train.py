#!/usr/bin/env python3
"""Train the System-1 motion substrate (system1_motion_spec.md §7).

A1-minimal by default: primary action-conditioned prediction + variance hinge +
DETACHED position-decode diagnostic. Temporal contrast (Aux 3) is off (--alpha 0)
and slots in as an ablation flag. Periodic Gate-0 checks via the validated
harness (gates.gate0_precommit.decode_gate) give a training-time pass/fail trace,
with a hard kill if it isn't beating chance by the spec's step-5000 checkpoint.

Reads a cached dataset (system1_motion/render_dataset.py). Import-tested; the
real run needs the rendered dataset (headless MuJoCo) on a GPU pod.

    python -m system1_motion.train --data reacher_transitions.npz --steps 30000
"""
from __future__ import annotations

import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import DataLoader

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead, ema_update
from system1_motion.objective import RunningSigma, substrate_loss
from gates.gate0_precommit import decode_gate


def ema_momentum(step, total, m0=0.99, m1=0.9999):
    f = min(step / max(total, 1), 1.0)
    return m0 + (m1 - m0) * f


def beta_ramp(step, warmup=2000, beta=1.0):
    return beta * min(step / max(warmup, 1), 1.0)


def gate0_check(encoder, frames, pos, img_px, device, steps=5000, threshold_px=5.0, bs=512):
    # NOTE: no_grad wraps ONLY the embedding — decode_gate trains a fresh decoder
    # and needs grad enabled (a @torch.no_grad() on this whole fn breaks backward).
    encoder.eval()
    Z = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            x = torch.from_numpy(frames[i:i + bs]).float().to(device) / 255.0
            Z.append(encoder(x).cpu().numpy())
    encoder.train()
    Z = np.concatenate(Z)
    return decode_gate(Z, pos, img_px, steps=steps, threshold_px=threshold_px, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--frame-stack", type=int, default=1,
                    help="S frames stacked on channels so the encoder sees velocity; "
                         "A1=1 (failed Gate 0 on 2nd-order Reacher), A2=3")
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--d-z", type=int, default=384)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--dyn-depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-var", type=float, default=1.0)
    ap.add_argument("--gate-every", type=int, default=5000)
    ap.add_argument("--kill-step", type=int, default=5000)
    ap.add_argument("--kill-px", type=float, default=25.0)     # spec: must beat chance by 5k
    ap.add_argument("--threshold-px", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/system1_motion_A1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from system1_motion.data import TransitionDataset
    ds = TransitionDataset(args.data, frame_stack=args.frame_stack)
    img_px = ds.img_px
    eval_frames, eval_pos = ds.eval_arrays(frac=0.2)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    da = ds.actions.shape[1]
    in_ch = 3 * args.frame_stack

    enc = ViTEncoder(args.image_size, args.patch, in_ch, args.d_z, args.enc_depth).to(dev)
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    dyn = LatentDynamics(args.d_z, da, args.dyn_depth).to(dev)
    dec = DecodeHead(args.d_z, out_dim=eval_pos.shape[1]).to(dev)
    opt_sub = torch.optim.AdamW(list(enc.parameters()) + list(dyn.parameters()), lr=args.lr, weight_decay=1e-4)
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    sigma = RunningSigma()

    trace, t0, step = [], time.time(), 0
    di = iter(dl)
    while step < args.steps:
        try:
            xt, a, xtp1, pos = next(di)
        except StopIteration:
            di = iter(dl); xt, a, xtp1, pos = next(di)
        xt, a, xtp1, pos = xt.to(dev), a.to(dev), xtp1.to(dev), pos.to(dev)

        z_t = enc(xt)
        with torch.no_grad():
            z_tgt = tgt(xtp1); sigma.update(z_tgt)
        z_pred = dyn(z_t, a)
        L_sub, parts = substrate_loss(z_pred, z_tgt, z_t, sigma.sigma2(args.d_z, dev),
                                      beta_var=beta_ramp(step, beta=args.beta_var))
        opt_sub.zero_grad(set_to_none=True); L_sub.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dyn.parameters()), 1.0)
        opt_sub.step()
        ema_update(tgt, enc, ema_momentum(step, args.steps))

        # detached position-decode diagnostic (separate optimizer; never trains substrate)
        p_pred = dec(z_t.detach())
        L_dec = ((p_pred - pos) ** 2).mean()
        opt_dec.zero_grad(set_to_none=True); L_dec.backward(); opt_dec.step()

        step += 1
        if step % 200 == 0:
            print(f"[step {step}/{args.steps}] pred={parts['pred']:.4f} var={parts['var']:.4f} "
                  f"decode_aux={float(L_dec):.3f} ({time.time()-t0:.0f}s)", flush=True)

        if step % args.gate_every == 0 or step == args.steps:
            g = gate0_check(enc, eval_frames, eval_pos, img_px, dev, threshold_px=args.threshold_px)
            g["step"] = step
            trace.append(g)
            print(f"[GATE0 @ step {step}] mean_l2={g['mean_l2_px']}px "
                  f"{'PASS' if g['pass'] else 'FAIL'} (threshold {args.threshold_px}px)", flush=True)
            torch.save({"enc": enc.state_dict(), "dyn": dyn.state_dict(), "args": vars(args)},
                       os.path.join(args.out_dir, "substrate.pt"))
            json.dump({"trace": trace, "final": trace[-1]},
                      open(os.path.join(args.out_dir, "gate0_trace.json"), "w"), indent=2)
            if step == args.kill_step and g["mean_l2_px"] > args.kill_px:
                print(f"[KILL] step {step}: decode {g['mean_l2_px']}px > {args.kill_px}px — "
                      "not beating chance; stopping per spec.", flush=True)
                break

    print(json.dumps({"final_gate0": trace[-1] if trace else None,
                      "out_dir": args.out_dir}, indent=2))


if __name__ == "__main__":
    main()
