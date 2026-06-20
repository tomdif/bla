#!/usr/bin/env python3
"""Closed-loop USABILITY probe for the grounding-dissociation rig.

Ports the matched-capacity closed-loop usability harness (synthetic origin:
~/grounding_meter) onto the real Reacher dissociation rig. It answers the one
question action-cosine and decode_gate cannot: is the UNCONTROLLED target
*usable* — can a controller reach it through the frozen latent — not merely
*decodable*.

Method (frozen substrate, fresh readouts — same philosophy as decode_gate):
  1. load a dissoc encoder checkpoint (substrate_*_{C0,C1,C2}_s{seed}.pt), freeze it;
  2. embed the pre-rendered dataset -> latents Z_t, Z_{t+1}, action, finger, target;
  3. fit FRESH readouts on the frozen latents: dyn(z,a)->z' and dec_ee(z)->finger;
  4. train three matched-capacity reach policies in latent IMAGINATION (grad through
     the frozen dyn — no mujoco at probe time):
        oracle : pi([z, g=true_target]) -> a      (upper bound: target known)
        latent : pi([z]) -> a                      (target available ONLY via z)
        blind  : pi([z]) -> a, trained on SHUFFLED (z,target) pairs (z carries no
                 target info; matched architecture to `latent`)  (lower bound)
  5. reach error (final imagined ||dec_ee(z_H) - target||);
     USABILITY(target) = (blind_err - latent_err) / (blind_err - oracle_err) in [~0,1].

Pre-registered dissociation (the real-rig test of "decode lies" for the UNCONTROLLED var):
  C2 (target-grounded): decode(target) PASS  AND usability(target) HIGH   -> decode==usable
  C1 (action-grounded): decode(target) FAIL  AND usability(target) LOW    -> the dissociation
  C0 (baseline)       : both low
  If C2 gives decode-PASS but usability-LOW -> phase-memory for the uncontrolled variable.

Run on a GPU pod (dataset + checkpoints live there):
  python -m system1_motion.usability_probe --ckpt runs/dissoc/substrate_pool_C1_s0.pt \
      --data runs/reacher_transitions.npz
Local wiring/logic check (no mujoco, no checkpoint):
  python -m system1_motion.usability_probe --smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn


# ----------------------------- frozen encoder -----------------------------
def load_encoder(ckpt_path, in_ch, device):
    from system1_motion.models import ViTEncoder
    ck = torch.load(ckpt_path, map_location=device)
    a = ck.get("args", {})
    enc_kind = a.get("encoder", "pool")
    if enc_kind == "slot":
        from system1_motion.slot_encoder import SlotEncoder
        enc = SlotEncoder(in_channels=in_ch, vit_dim=a["vit_dim"], patch=a["patch"],
                          depth=a["enc_depth"], n_slots=a["n_slots"], slot_dim=a["slot_dim"])
    else:
        enc = ViTEncoder(a.get("image_size", 64), a.get("patch", 8), in_ch,
                         a.get("d_z", 384), a.get("enc_depth", 6))
    enc.load_state_dict(ck["enc"]); enc.to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    d_z = a.get("d_z", None) or next(enc.parameters()).shape  # fallback
    return enc, a


@torch.no_grad()
def embed_dataset(enc, data_path, device, max_n=20000, bs=512):
    from system1_motion.data import TransitionDataset
    ds = TransitionDataset(data_path, frame_stack=1, return_target=True)
    if ds.target is None:
        raise SystemExit("dataset has no 'target' key — re-render with target extraction.")
    pairs = ds.pairs[:max_n]
    fr = ds.frames
    def emb(idxs):
        Z = []
        for i in range(0, len(idxs), bs):
            chunk = idxs[i:i + bs]
            x = torch.from_numpy(fr[chunk].reshape(len(chunk), *fr.shape[2:])).float().to(device) / 255.0
            Z.append(enc(x).cpu())
        return torch.cat(Z)
    Zt = emb(pairs)
    Ztp1 = emb(pairs + 1)
    A = torch.from_numpy(ds.actions[pairs].astype(np.float32))
    finger = torch.from_numpy(ds.pos[pairs].astype(np.float32))
    finger_next = torch.from_numpy(ds.pos[pairs + 1].astype(np.float32))
    target = torch.from_numpy(ds.target[pairs].astype(np.float32))
    return Zt, A, Ztp1, finger, finger_next, target, ds.img_px


def _lin_r2(Z, Y, seed=0, split=0.8):
    """Linear-readout R^2 of Y from latent Z (held-out)."""
    Z = np.asarray(Z, np.float32); Y = np.asarray(Y, np.float32)
    n = len(Z); ntr = int(split * n)
    A = np.concatenate([Z[:ntr], np.ones((ntr, 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(A, Y[:ntr], rcond=None)
    P = np.concatenate([Z[ntr:], np.ones((n - ntr, 1), np.float32)], 1) @ W
    Yt = Y[ntr:]
    ss = ((Yt - P) ** 2).sum(); tot = ((Yt - Yt.mean(0)) ** 2).sum()
    return float(max(0.0, 1.0 - ss / max(tot, 1e-9)))


def arm_shortcut_probe(Zt, Ztp1, finger, finger_next):
    """Pred-3 diagnostic: does the encoder ground ABSOLUTE finger position (in each latent)
    or only the DELTA (recoverable from the latent TRANSITION)? Inverse-dynamics grounds the
    action-quotient => arm_vel_r2 >> arm_abs_r2 ('grounds the controllable quotient, not the
    controllable state' = why ID was dead). Decode/vanilla-substrate ground both.
      arm_abs_r2 = R^2( finger[t]            | Z_t )
      arm_vel_r2 = R^2( finger[t+1]-finger[t]| [Z_t, Z_{t+1}] )   (the ID transition signal)"""
    Zt, Ztp1 = Zt.numpy(), Ztp1.numpy()
    fa = finger.numpy(); fv = (finger_next - finger).numpy()
    return {"arm_abs_r2": _lin_r2(Zt, fa),
            "arm_vel_r2": _lin_r2(np.concatenate([Zt, Ztp1], 1), fv)}


# ----------------------------- fresh readouts on frozen latents -----------------------------
def fit_dynamics(Zt, A, Ztp1, d_a, device, steps=4000, bs=256, lr=3e-4):
    from system1_motion.models import LatentDynamics
    d_z = Zt.shape[1]
    dyn = LatentDynamics(d_z, d_a, depth=4).to(device)
    opt = torch.optim.AdamW(dyn.parameters(), lr=lr)
    Zt, A, Ztp1 = Zt.to(device), A.to(device), Ztp1.to(device)
    n = len(Zt)
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        loss = ((dyn(Zt[idx], A[idx]) - Ztp1[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in dyn.parameters(): p.requires_grad_(False)
    dyn.eval()
    with torch.no_grad():
        resid = ((dyn(Zt, A) - Ztp1) ** 2).mean().item()
        var = Ztp1.var().item()
    return dyn, 1.0 - resid / max(var, 1e-9)   # dyn, R^2


def fit_ee_decoder(Zt, finger, device, steps=4000, bs=256, lr=3e-4):
    from system1_motion.models import DecodeHead
    dec = DecodeHead(Zt.shape[1], out_dim=finger.shape[1]).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr)
    Zt, finger = Zt.to(device), finger.to(device); n = len(Zt)
    for it in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        loss = ((dec(Zt[idx]) - finger[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in dec.parameters(): p.requires_grad_(False)
    dec.eval()
    with torch.no_grad():
        err = ((dec(Zt) - finger) ** 2).sum(-1).sqrt().mean().item()
    return dec, err


# ----------------------------- reach policy (matched capacity) -----------------------------
class ReachPolicy(nn.Module):
    def __init__(self, d_z, d_a, g_dim=0, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_z + g_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d_a))
        with torch.no_grad(): self.net[-1].weight.mul_(0.05); self.net[-1].bias.mul_(0.)

    def forward(self, z, g=None):
        return torch.tanh(self.net(z if g is None else torch.cat([z, g], -1)))


def real_rollout_eval(enc, pol, mode, device, seed=0, episodes=150, H=30,
                      action_repeat=8, extent=0.27, img=64, task="easy", camera_id=0):
    """ARBITER: evaluate a policy in the ACTUAL Reacher physics (no fitted dyn in the
    verdict). Policy was trained cheaply in imagination; rolling it in the real sim
    is the lie-detector for imagination (WM-exploitation -> low imagined err but high
    real err). px units match render_dataset (== the dataset's target/pos fields).
    POD-ONLY (needs dm_control + mujoco + MUJOCO_GL)."""
    from dm_control import suite
    import mujoco
    env = suite.load("reacher", task, task_kwargs={"random": seed})
    renderer = mujoco.Renderer(env.physics.model.ptr, height=img, width=img)
    aspec = env.action_spec()
    px = lambda name: (np.asarray(env.physics.named.data.geom_xpos[name][:2]) + extent) / (2 * extent) * img
    pol.eval(); errs = []
    with torch.no_grad():
        for ep in range(episodes):
            env.reset(); g_px = px("target").astype(np.float32)
            g = torch.tensor(g_px, device=device)[None] if mode == "oracle" else None
            for t in range(H):
                renderer.update_scene(env.physics.data.ptr, camera=camera_id)
                x = torch.from_numpy(renderer.render().copy()).permute(2, 0, 1)[None].float().to(device) / 255.0
                a = pol(enc(x), g).cpu().numpy()[0]
                a = np.clip(a, aspec.minimum, aspec.maximum)
                done = False
                for _ in range(action_repeat):
                    if env.step(a).last(): done = True; break
                if done: break
            errs.append(float(np.linalg.norm(px("finger") - g_px)))
    return float(np.mean(errs))


def train_reach(mode, Z, target, dyn, dec_ee, d_a, device,
                H=10, steps=2500, bs=256, gamma=0.95, areg=0.02, lr=3e-4, seed=0,
                ens=None, dis_beta=0.0):
    """mode: 'oracle' (g=true target), 'latent' (no g, target only via z),
    'blind' (no g, trained on SHUFFLED (z,target) -> z useless for target).
    ens: optional list of extra dyn heads; dis_beta>0 adds PETS/MBPO disagreement
    penalty (cost += dis_beta * std_k ee_k) to stop the policy exploiting imagined
    dynamics — the no-sim robustification when real-rollout is unavailable.
    Returns (eval_reach_err, trained_policy)."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    Z, target = Z.to(device), target.to(device); n = len(Z)
    g_dim = target.shape[1] if mode == 'oracle' else 0
    pol = ReachPolicy(Z.shape[1], d_a, g_dim).to(device)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr)
    ntr = int(0.8 * n); tr = np.arange(ntr); te = np.arange(ntr, n)
    for it in range(steps):
        idx = torch.from_numpy(rng.choice(tr, bs)).to(device)
        z = Z[idx]
        tgt = target[idx]
        if mode == 'blind':                                  # decorrelate target from z (shuffle control)
            tgt = target[torch.from_numpy(rng.choice(tr, bs)).to(device)]
        g = tgt if mode == 'oracle' else None
        zc, cost = z, 0.0
        for h in range(H):
            a = pol(zc, g)
            if ens and dis_beta > 0:                         # PETS/MBPO: penalize exploiting imagined dynamics
                ees = torch.stack([dec_ee(m(zc, a)) for m in [dyn] + ens], 0)   # [K+1,B,2]
                cost = cost + dis_beta * ees.std(0).sum(-1).mean()
            zc = dyn(zc, a)
            cost = cost + (gamma ** h) * ((dec_ee(zc) - tgt) ** 2).sum(-1).mean()
            cost = cost + areg * (a ** 2).mean()
        opt.zero_grad(); cost.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
    # eval on held-out with TRUE (z,target) pairing for all modes (blind just lacks the info in its weights)
    pol.eval()
    with torch.no_grad():
        idx = torch.from_numpy(te).to(device); z = Z[idx]; tgt = target[idx]
        g = tgt if mode == 'oracle' else None
        zc = z
        for h in range(H):
            a = pol(zc, g); zc = dyn(zc, a)
        err = ((dec_ee(zc) - tgt) ** 2).sum(-1).sqrt().mean().item()
    return err, pol


def usability_target(Z, target, dyn, dec_ee, d_a, device, real_eval=None, **kw):
    eo, po = train_reach('oracle', Z, target, dyn, dec_ee, d_a, device, **kw)
    el, pl = train_reach('latent', Z, target, dyn, dec_ee, d_a, device, **kw)
    eb, pb = train_reach('blind',  Z, target, dyn, dec_ee, d_a, device, **kw)
    out = {"oracle_err_imag": eo, "latent_err_imag": el, "blind_err_imag": eb,
           "usability_target_imag": (eb - el) / (eb - eo + 1e-9)}
    if real_eval is not None:                                 # ARBITER: re-evaluate the SAME policies in real physics
        ro = real_eval(po, 'oracle'); rl = real_eval(pl, 'latent'); rb = real_eval(pb, 'blind')
        out.update({"oracle_err_real": ro, "latent_err_real": rl, "blind_err_real": rb,
                    "usability_target_real": (rb - rl) / (rb - ro + 1e-9)})
    return out


def fit_ensemble(Zt, A, Ztp1, d_a, device, K=3, steps=2500):
    """Extra dyn heads (diff seeds) for the imagination disagreement penalty."""
    heads = []
    for k in range(K):
        torch.manual_seed(100 + k)
        dyn, _ = fit_dynamics(Zt, A, Ztp1, d_a, device, steps=steps)
        heads.append(dyn)
    return heads


# ----------------------------- driver -----------------------------
def run_condition(ckpt, data, device, max_n, in_ch=3, real_rollout=False, ens_K=0, dis_beta=0.0):
    from gates.gate0_precommit import decode_gate
    enc, a = load_encoder(ckpt, in_ch, device)
    Zt, A, Ztp1, finger, finger_next, target, img_px = embed_dataset(enc, data, device, max_n=max_n)
    d_a = A.shape[1]
    dyn, dyn_r2 = fit_dynamics(Zt, A, Ztp1, d_a, device)
    dec_ee, ee_err = fit_ee_decoder(Zt, finger, device)
    dt = decode_gate(Zt.numpy(), target.numpy(), img_px, device=device)   # decode lie-detector
    arm = arm_shortcut_probe(Zt, Ztp1, finger, finger_next)               # pred-3 abs-vs-vel diagnostic
    ens = fit_ensemble(Zt, A, Ztp1, d_a, device, K=ens_K) if ens_K > 0 else None
    real_eval = (lambda pol, mode: real_rollout_eval(enc, pol, mode, device, img=img_px)) if real_rollout else None
    res = usability_target(Zt, target, dyn, dec_ee, d_a, device,
                           real_eval=real_eval, ens=ens, dis_beta=dis_beta)
    res.update({"decode_target_px": dt.get("mean_px"), "decode_target_pass": dt.get("passed"),
                "arm_abs_r2": arm["arm_abs_r2"], "arm_vel_r2": arm["arm_vel_r2"],
                "dyn_r2": dyn_r2, "ee_decode_err_px": ee_err, "ens_K": ens_K, "dis_beta": dis_beta,
                "condition": a.get("condition", "?"), "seed": a.get("seed", "?")})
    return res


# ----------------------------- smoke (CPU, no mujoco/ckpt) -----------------------------
def smoke():
    """Validate the full pipeline logic on a synthetic structured latent world, AND
    that the real ViTEncoder constructs + runs on a dummy frame."""
    dev = "cpu"; torch.manual_seed(0); np.random.seed(0)
    print("[smoke] (A) pipeline logic on synthetic structured latents")
    d_z, d_a, N = 24, 2, 4000
    # latent: dims[0:2]=finger, dims[2:4]=target (per-episode const), rest noise
    finger = np.random.randn(N, 2).astype(np.float32)
    target = np.random.randn(N, 2).astype(np.float32)
    A = np.random.uniform(-1, 1, (N, d_a)).astype(np.float32)
    finger1 = finger + 0.2 * A                                   # controllable dynamics
    def Zof(fg, tg, encode_target):
        noise = 0.1 * np.random.randn(N, d_z - 4).astype(np.float32)
        tt = tg if encode_target else 0 * tg
        return np.concatenate([fg, tt, noise], 1).astype(np.float32)
    for encode_target, label in [(True, "target ENCODED"), (False, "target ABSENT")]:
        Zt = torch.tensor(Zof(finger, target, encode_target))
        Ztp1 = torch.tensor(Zof(finger1, target, encode_target))
        dyn, r2 = fit_dynamics(Zt, torch.tensor(A), Ztp1, d_a, dev, steps=1500)
        dec_ee, ee = fit_ee_decoder(Zt, torch.tensor(finger), dev, steps=1500)
        r = usability_target(Zt, torch.tensor(target), dyn, dec_ee, d_a, dev, steps=1200, H=8)
        print(f"  [{label}] dyn_r2={r2:.2f} ee_err={ee:.2f} | oracle={r['oracle_err_imag']:.2f} "
              f"latent={r['latent_err_imag']:.2f} blind={r['blind_err_imag']:.2f} | USABILITY={r['usability_target_imag']:.2f}")
    print("  expect: target ENCODED -> usability high; target ABSENT -> usability ~0 (latent~blind)")
    print("\n[smoke] (A2) ensemble disagreement-penalty path (no-sim robustification)")
    Zt = torch.tensor(Zof(finger, target, True)); Ztp1 = torch.tensor(Zof(finger1, target, True))
    dyn, _ = fit_dynamics(Zt, torch.tensor(A), Ztp1, d_a, dev, steps=800)
    dec_ee, _ = fit_ee_decoder(Zt, torch.tensor(finger), dev, steps=800)
    ens = fit_ensemble(Zt, torch.tensor(A), Ztp1, d_a, dev, K=2, steps=600)
    r = usability_target(Zt, torch.tensor(target), dyn, dec_ee, d_a, dev, steps=600, H=6, ens=ens, dis_beta=1.0)
    print(f"  ensemble penalty path OK: usability_imag={r['usability_target_imag']:.2f} (K=2 heads, dis_beta=1.0)")
    print("\n[smoke] (A3) arm-shortcut diagnostic (ID grounds the DELTA, not absolute)")
    fa = torch.tensor(finger); fn = torch.tensor(finger1)                 # finger1 = finger + 0.2A
    nz = lambda: 0.1 * np.random.randn(N, d_z - 2).astype(np.float32)
    # decode-like: each latent carries absolute finger -> abs high AND delta recoverable from pair
    Zt_abs = torch.tensor(np.concatenate([finger,  nz()], 1)); Ztp1_abs = torch.tensor(np.concatenate([finger1, nz()], 1))
    # ID-like: an episode-constant offset c masks absolute finger, but cancels in the transition (clean delta)
    c = np.random.randn(N, 2).astype(np.float32) * 3.0
    Zt_id = torch.tensor(np.concatenate([finger + c,  nz()], 1)); Ztp1_id = torch.tensor(np.concatenate([finger1 + c, nz()], 1))
    a_dec = arm_shortcut_probe(Zt_abs, Ztp1_abs, fa, fn); a_id = arm_shortcut_probe(Zt_id, Ztp1_id, fa, fn)
    print(f"  decode-like : abs_r2={a_dec['arm_abs_r2']:.2f} vel_r2={a_dec['arm_vel_r2']:.2f}  (both high)")
    print(f"  ID-like     : abs_r2={a_id['arm_abs_r2']:.2f} vel_r2={a_id['arm_vel_r2']:.2f}  (vel>>abs = quotient shortcut)")
    print("\n[smoke] (B) real ViTEncoder constructs + runs on dummy frame")
    from system1_motion.models import ViTEncoder
    enc = ViTEncoder(64, 8, 3, 96, 2)  # d_z divisible by 6 heads
    z = enc(torch.zeros(2, 3, 64, 64))
    print(f"  ViTEncoder OK: input [2,3,64,64] -> latent {tuple(z.shape)} (real path uses d_z=384)")
    print("[smoke] wiring OK — pod run needs real --ckpt + --data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="dissoc substrate_*.pt (encoder + args)")
    ap.add_argument("--data", help="reacher_transitions.npz (must have 'target')")
    ap.add_argument("--out-dir", default="runs/dissoc")
    ap.add_argument("--max-n", type=int, default=20000)
    ap.add_argument("--real-rollout", action="store_true", help="ARBITER: eval policies in real Reacher physics (pod-only)")
    ap.add_argument("--ensemble", type=int, default=0, help="K extra dyn heads for disagreement penalty")
    ap.add_argument("--dis-beta", type=float, default=0.0, help="PETS disagreement penalty weight")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(); return
    if not (args.ckpt and args.data):
        ap.error("need --ckpt and --data (or --smoke)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    res = run_condition(args.ckpt, args.data, dev, args.max_n,
                        real_rollout=args.real_rollout, ens_K=args.ensemble, dis_beta=args.dis_beta)
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2))
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.ckpt))[0]
    json.dump(res, open(os.path.join(args.out_dir, f"usability_{base}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
