#!/usr/bin/env python3
"""Pure self-supervised multi-step LATENT rollout — the decisive test for whether
rollout prediction is a genuine grounding sponsor or was supervision in disguise.

The Layer-2 ranking claims rollout prediction grounds absolute position ("the
second OF-JEPA hit 2-3px because predicting forward demanded it"). But that pipeline
also had Hungarian-matched position supervision + a position-supervised rollout loss,
so the 2-3px credits SUPERVISION. This script removes ALL ground-truth decode from
the loss and asks: does pure latent rollout, by itself, put absolute position in z?

Loss (no GT anywhere):
    z_0 = enc(stack_t)                         # online
    for k in 0..H-1: z_{k+1} = dyn(z_k, E_a(a_{t+k}))   # roll in latent space
    L = mean_k normalized_mse(z_{k+1}, sg[ tgt(stack_{t+k+1}) ]) + beta_var * var_hinge(z_0)

Then a fresh linear probe decodes finger/target position from z_0 (DIAGNOSTIC, not in
the loss). Checkpoint is legibility.py-compatible so the same frozen battery scores it.

  rollout grounds position (arm px drops << 18) -> rollout is a real self-sup sponsor
  arm stays ~18px                               -> it was supervision all along; Layer 2
                                                   has ONE sponsor (explicit decode), not three

    python -m system1_motion.train_rollout_ss --data runs/reacher_transitions.npz \
        --encoder pool --rollout-h 8 --steps 30000
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead, ema_update
from system1_motion.slot_encoder import SlotEncoder
from system1_motion.objective import RunningSigma, RunningStandardizer, normalized_mse, variance_hinge
from system1_motion.aux_networks import ActionEmbedder
from gates.gate0_precommit import decode_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", default="pool", choices=["pool", "slot"])
    ap.add_argument("--condition", default="ROLL")
    ap.add_argument("--rollout-h", type=int, default=8)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--d-z", type=int, default=384)
    ap.add_argument("--d-u", type=int, default=32)
    ap.add_argument("--enc-depth", type=int, default=6)
    ap.add_argument("--dyn-depth", type=int, default=4)
    ap.add_argument("--vit-dim", type=int, default=192)
    ap.add_argument("--n-slots", type=int, default=6)
    ap.add_argument("--slot-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta-var", type=float, default=1.0)
    ap.add_argument("--gate-every", type=int, default=5000)
    ap.add_argument("--threshold-px", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/rollout_ss")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device if (torch.cuda.is_available() or args.device in ("cpu", "mps")) else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)
    S, H = args.frame_stack, args.rollout_h
    print(f"[rollout-ss] {args.encoder} H={H} S={S} cond={args.condition} dev={dev}", flush=True)

    d = np.load(args.data)
    frames = d["frames"]; actions = d["actions"].astype(np.float32)
    pos = d["pos"].astype(np.float32); target = d["target"].astype(np.float32)
    ep = d["ep_id"].astype(np.int64); img_px = int(d["img_px"]); N = len(frames); HW = frames.shape[2]
    da = actions.shape[1]
    # valid window bases t: frames[t-S+1 .. t+H] all in-episode
    bases = np.array([t for t in range(S - 1, N - H) if ep[t - S + 1] == ep[t + H]])
    rng = np.random.RandomState(0); rng.shuffle(bases)
    nval = min(800, max(1, len(bases) // 5)); val_b, tr_b = bases[:nval], bases[nval:]
    print(f"[rollout-ss] valid window bases: {len(bases)} (val {nval}, tr {len(tr_b)})", flush=True)

    def stacks_at(t_arr, k):
        """[B, 3S, HW, HW] : the S-stack ending at frame t+k for each base t."""
        out = np.empty((len(t_arr), 3 * S, HW, HW), np.uint8)
        for j, t in enumerate(t_arr):
            out[j] = frames[t + k - S + 1: t + k + 1].reshape(-1, HW, HW)
        return out

    in_ch = 3 * S
    if args.encoder == "slot":
        enc = SlotEncoder(in_channels=in_ch, vit_dim=args.vit_dim, patch=args.patch,
                          depth=args.enc_depth, n_slots=args.n_slots, slot_dim=args.slot_dim).to(dev)
        args.d_z = enc.d_z
    else:
        enc = ViTEncoder(args.image_size, args.patch, in_ch, args.d_z, args.enc_depth).to(dev)
    tgt = copy.deepcopy(enc).to(dev)
    for p in tgt.parameters():
        p.requires_grad_(False)
    dyn = LatentDynamics(args.d_z, args.d_u, args.dyn_depth).to(dev)
    E_a = ActionEmbedder(da, args.d_u).to(dev)
    dec = DecodeHead(args.d_z, out_dim=2).to(dev)            # detached probe only
    params = list(enc.parameters()) + list(dyn.parameters()) + list(E_a.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    opt_dec = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    sigma = RunningSigma(); astd = RunningStandardizer()

    def ema_m(step): return 0.99 + (0.9999 - 0.99) * min(step / args.steps, 1.0)

    @torch.no_grad()
    def embed_base(enc_, t_arr, bs=256):
        enc_.eval(); Z = []
        for i in range(0, len(t_arr), bs):
            x = torch.from_numpy(stacks_at(t_arr[i:i+bs], 0)).float().to(dev) / 255.0
            Z.append(enc_(x).cpu().numpy())
        enc_.train(); return np.concatenate(Z)

    def probes(step):
        Z = embed_base(enc, val_b)
        g_a = decode_gate(Z, pos[val_b], img_px, steps=5000, threshold_px=args.threshold_px, device=dev)
        g_t = decode_gate(Z, target[val_b], img_px, steps=5000, threshold_px=args.threshold_px, device=dev)
        print(f"[PROBE @ {step}] {args.encoder}/{args.condition} | ARM={g_a['mean_l2_px']}px "
              f"{'PASS' if g_a['pass'] else 'FAIL'} | TARGET={g_t['mean_l2_px']}px "
              f"{'PASS' if g_t['pass'] else 'FAIL'}", flush=True)
        return {"step": step, "arm_px": g_a["mean_l2_px"], "arm_pass": g_a["pass"],
                "target_px": g_t["mean_l2_px"], "target_pass": g_t["pass"]}

    trace, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        sel = tr_b[np.random.randint(0, len(tr_b), args.batch)]
        x0 = torch.from_numpy(stacks_at(sel, 0)).float().to(dev) / 255.0
        z = enc(x0)
        # standardize actions over the window
        aw = torch.from_numpy(np.stack([actions[sel + k] for k in range(H)], 1)).to(dev)  # [B,H,da]
        astd.update(aw.reshape(-1, da))
        L = z.new_zeros(())
        for k in range(H):
            u = E_a(astd(aw[:, k]))
            z = dyn(z, u)
            with torch.no_grad():
                xk = torch.from_numpy(stacks_at(sel, k + 1)).float().to(dev) / 255.0
                zt = tgt(xk); sigma.update(zt)
            L = L + normalized_mse(z, zt, sigma.sigma2(args.d_z, dev))
        L = L / H + args.beta_var * variance_hinge(enc(x0))
        opt.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        ema_update(tgt, enc, ema_m(step))

        with torch.no_grad():
            z0d = enc(x0)
        L_dec = ((dec(z0d.detach()) - torch.from_numpy(pos[sel]).to(dev)) ** 2).mean()
        opt_dec.zero_grad(set_to_none=True); L_dec.backward(); opt_dec.step()

        if step % 200 == 0:
            print(f"[step {step}/{args.steps}] roll_pred={float(L):.4f} dec_aux={float(L_dec):.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step % args.gate_every == 0 or step == args.steps:
            trace.append(probes(step))
            torch.save({"enc": enc.state_dict(), "args": vars(args)},
                       os.path.join(args.out_dir, f"substrate_{args.condition}_{args.encoder}_s{args.seed}.pt"))
            json.dump({"trace": trace, "final": trace[-1]},
                      open(os.path.join(args.out_dir, f"trace_{args.condition}_{args.encoder}_s{args.seed}.json"), "w"), indent=2)

    print(json.dumps({"encoder": args.encoder, "final": trace[-1] if trace else None}, indent=2))


if __name__ == "__main__":
    main()
