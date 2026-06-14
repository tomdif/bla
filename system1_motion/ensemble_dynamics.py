#!/usr/bin/env python3
"""Ensemble dynamics -- the convergent mechanism for exploitation-robust planning. Reuse the FROZEN encoder +
decoder from the torque world model; train K independent latent-dynamics heads (different init + bootstrap data
order). Their DISAGREEMENT on a (z, a) is epistemic uncertainty: low where data is dense / actions are in-dist,
HIGH where a learned proposer exploits model error.

Two uses (both validated here): (1) DETECT exploitation for the governor (reject high-disagreement proposals),
(2) PENALIZE exploitation in planning (cost += beta * disagreement; PETS/MBPO style).

KEY VALIDATION first: does disagreement separate the POLICY's exploited actions from IN-DIST actions? If yes the
mechanism works. Run: MUJOCO_GL=egl python3 -m system1_motion.ensemble_dynamics --train --validate
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from system1_motion.models import LatentDynamics
from system1_motion.r3_torque import Arm, load_wm3d, collect_exploration, norm3, SPAN, ADIM, IMG
import system1_motion.r3_torque as R

CKPT = "runs/r3t_ckpt/wm3dt.pt"; ENS = "runs/r3t_ckpt/ensemble.pt"
span_t = None


@torch.no_grad()
def encode_all(enc, frames, device, bs=512):
    fr = torch.from_numpy(frames); out = []
    for i in range(0, len(fr), bs):
        out.append(enc(fr[i:i + bs].float().to(device) / 255.0))
    return torch.cat(out)


def train_ensemble(wm, trans, device, K=3, steps=5000, cons_k=4, cons_w=15.0, lr=3e-4, batch=128, log=print):
    F_, A, P, T, E = trans; idx = np.where(E[:-1] == E[1:])[0]
    fr_a = torch.from_numpy(A).to(device); fr_p = torch.from_numpy(P).to(device)
    log("[ens] encoding frames with frozen encoder ...", flush=True)
    Z = encode_all(wm["enc"], F_, device)                     # [N,384] frozen latents
    idxset = set(int(s) for s in idx)
    kstarts = np.array([s for s in idx if all((int(s) + k) in idxset for k in range(cons_k))])
    dec_g = wm["dec_g"]
    for p in dec_g.parameters(): p.requires_grad_(False)
    heads = []
    for k in range(K):
        torch.manual_seed(100 + k)
        dyn = LatentDynamics(384, ADIM, 4).to(device); opt = torch.optim.AdamW(dyn.parameters(), lr=lr)
        brng = np.random.RandomState(k); t0 = time.time()
        for step in range(steps):
            b = brng.choice(kstarts, batch)                   # bootstrap (different order per head)
            z0, z1, a = Z[b], Z[b + 1], fr_a[b]
            pred = F.mse_loss(dyn(z0, a), z1)
            zr = z0; cons = 0.0
            for j in range(cons_k):
                zr = dyn(zr, fr_a[b + j]); cons = cons + F.mse_loss(dec_g(zr), fr_p[b + j + 1])
            loss = pred + cons_w * cons / cons_k
            opt.zero_grad(); loss.backward(); opt.step()
            if step % max(1, steps // 4) == 0 or step == steps - 1:
                log(f"[ens head{k} {step}/{steps}] pred={pred.item():.4f} cons={cons.item()/cons_k:.4f} ({time.time()-t0:.0f}s)", flush=True)
        heads.append(dyn.eval())
    return heads


@torch.no_grad()
def disagreement_cm(heads, dec_g, z, a):
    """spread (cm) of the K heads' decoded next-ee for (z,a). z[B,384], a[B,adim] -> [B] cm."""
    ees = torch.stack([dec_g(h(z, a)) for h in heads])        # [K,B,3] normalized
    spread = ((ees - ees.mean(0, keepdim=True)) * span_t).pow(2).sum(-1).sqrt().mean(0)   # [B] meters
    return spread * 100


def load_ensemble(path, device):
    sds = torch.load(path, map_location=device)["heads"]; heads = []
    for sd in sds:
        d = LatentDynamics(384, ADIM, 4).to(device); d.load_state_dict(sd); d.eval(); heads.append(d)
    return heads


def validate(wm, heads, device, log=print):
    """KEY TEST: disagreement on IN-DIST vs RANDOM vs POLICY-EXPLOITED actions. Exploited >> in-dist => works."""
    from system1_motion.imagination_policy import ImaginationPolicy, zr_norm
    arm = Arm(123); rng = np.random.RandomState(0)
    Zs, A_ind = [], []
    for _ in range(600):                                      # in-dist (z, a): pd-reach actions from real states
        arm.reset(); arm.set_target(arm.sample_target())
        x = torch.from_numpy(arm.render().astype(np.float32) / 255.0)[None].to(device)
        Zs.append(wm["enc"](x)); A_ind.append(arm.pd_reach())
    Z = torch.cat(Zs); A_ind = torch.tensor(np.array(A_ind), device=device).float()
    A_rand = torch.empty_like(A_ind).uniform_(-1, 1)
    # policy-exploited actions
    pol = ImaginationPolicy().to(device)
    pol.load_state_dict(torch.load("runs/r3t_ckpt/policy.pt", map_location=device)["state"]); pol.eval()
    G = torch.tensor(np.array([norm3(Arm(0).sample_target()) for _ in range(Z.shape[0])]), device=device).float()
    zone = torch.zeros(Z.shape[0], 4, device=device)
    with torch.no_grad(): A_pol = pol(Z, G, zone)
    d_ind = disagreement_cm(heads, wm["dec_g"], Z, A_ind).mean().item()
    d_rand = disagreement_cm(heads, wm["dec_g"], Z, A_rand).mean().item()
    d_pol = disagreement_cm(heads, wm["dec_g"], Z, A_pol).mean().item()
    arm.close()
    log(f"[ens] DISAGREEMENT (cm):  in-dist={d_ind:.2f}  random={d_rand:.2f}  policy-exploited={d_pol:.2f}", flush=True)
    sep = d_pol / max(1e-6, d_ind)
    log(f"[ens] separation policy/in-dist = {sep:.2f}x  -> {'WORKS (disagreement flags exploitation)' if sep > 1.6 else 'WEAK (ensemble does not separate exploitation)'}", flush=True)
    return {"in_dist": d_ind, "random": d_rand, "policy": d_pol, "sep": sep}


def main():
    global span_t
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true"); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--explore", type=int, default=16000)
    args = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    R.AR = 12; span_t = torch.tensor(SPAN, device=dev).float()
    if args.smoke: args.steps, args.explore = 300, 2000
    wm = load_wm3d(CKPT, dev); print(f"[ens] loaded WM (grip_cm={wm['grip_cm']:.2f}) K={args.K}", flush=True)
    heads = None
    if args.train or args.smoke:
        print("[ens] collecting exploration (action-coverage) ...", flush=True)
        trans = collect_exploration(args.explore, ep_len=28, rand_frac=0.65)
        heads = train_ensemble(wm, trans, dev, K=args.K, steps=args.steps)
        torch.save({"heads": [h.state_dict() for h in heads]}, ENS); print(f"[ens] saved {args.K} heads -> {ENS}", flush=True)
    if args.validate or args.smoke:
        if heads is None: heads = load_ensemble(ENS, dev)
        validate(wm, heads, dev)


if __name__ == "__main__":
    main()
