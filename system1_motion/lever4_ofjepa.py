#!/usr/bin/env python3
"""LEVER 4 -- OF-JEPA COMPOSITIONAL DYNAMICS: sample-efficiency of DYNAMICS learning.

THE BET ----------------------------------------------------------------------
The R1 world-model planning win comes from DYNAMICS COVERAGE (state x action),
not goal coverage. The architectural hypothesis tested here: an OBJECT-FACTORIZED
(slot / OF-JEPA) world model learns good dynamics from LESS interaction data than a
monolithic (mean-pool ViT) world model, because slots factor the scene and let the
dynamics model generalize compositionally. If true, the slot WM should reach a
target planning success at a SMALLER data budget D than the monolithic WM.

We sweep the data budget D (subsample D transitions from the SAME exploration npz),
D in {2000, 8000, 30000, full}, and for EACH (encoder in {pool, slot}, D) train a
world model with the IDENTICAL validated R1 anti-collapse loss recipe + identical
steps, then eval CEM planning success on TRAIN goals (thresholds 6, 12 px).

>>> HONESTY / SCOPE CAVEAT (READ THIS) <<<
Reacher is a SINGLE-OBJECT scene: ONE 2-link arm + ONE static target. Compositional
/ object-factorization gains are EXPECTED TO BE WEAK OR ABSENT here, because the real
OF-JEPA payoff is in MULTI-object scenes -- "learn one object's dynamics, predict N
objects" -- which a single arm cannot exercise. A NULL result on Reacher is therefore
INFORMATIVE, not a failure of the idea: it says "the slot prior buys little on a
single-object task; the proper test needs a multi-object env." We PRE-REGISTER that
reading. A genuine WIN here would be a (mild) surprise -- slot's spatial structure /
absolute-position patches grounding dynamics from less data even on one object. A
multi-object variant (sketched in `multi_object_env_sketch` at the bottom) is the
REAL test of the compositional claim; this script delivers the honest single-object
baseline + the sweep machinery that the multi-object test would reuse verbatim.

PRE-REGISTERED GATE (set BEFORE running) -- printed OK/XX + VERDICT + json:
  (E) SAMPLE-EFFICIENCY-OF-DYNAMICS WIN: the slot/OF-JEPA WM reaches succ@12 >= 0.70
      at a SMALLER data budget D* than the monolithic WM, specifically
      D*_slot <= 0.5 x D*_pool   (D* = smallest D in the sweep clearing succ@12>=0.70).
  (C) FULL-D SANITY: at the full budget both encoders reach comparable succ@12
      (|succ_slot - succ_pool| <= 0.15) -- slot isn't just globally better/worse for
      unrelated (capacity/optimization) reasons; the claim is about the D-CURVE shape.
  FALSIFICATION: if slot needs D*_slot >= D*_pool (no smaller budget suffices), there
      is NO compositional sample-efficiency advantage on this single-object task ->
      verdict NULL, and the informative next step is the multi-object env.

REUSE: identical loss recipe / planner / eval as system1_motion.r1_imitation_fails
(plain-MSE latent prediction + stop-grad target + 5x supervised normalized [0,1]
fingertip+target decode grounding + variance_hinge; dec_arm/dec_tgt -> normalized
[0,1]; cem_plan expects a normalized target). The ONLY change vs train_world_model is
an ENCODER CHOICE (ViTEncoder vs SlotEncoder) -- everything downstream is identical so
any difference is attributable to the encoder. Does NOT modify r1_imitation_fails.py
or slot_encoder.py.
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead
from system1_motion.slot_encoder import SlotEncoder
from system1_motion.objective import variance_hinge
from system1_motion.r1_imitation_fails import (
    load_transitions, gen_expert_demos, cem_plan, eval_method, EXTENT,
)

# d_z=384 = 6 slots x 64 slot_dim = divisible by heads=6 (LatentDynamics/DecodeHead unchanged).
D_Z = 384
N_SLOTS, SLOT_DIM = 6, 64


def build_encoder(kind, img, in_ch=3, patch=8, depth=6):
    """Encoder factory -- the ONLY thing that differs between the two arms.
    'pool' = monolithic mean-pool ViTEncoder([B,3,H,W]->[B,384]).
    'slot' = OF-JEPA SlotEncoder([B,3,H,W]->[B, n_slots*slot_dim = 384]); its forward
    already FLATTENS slots [B,K,slot_dim]->[B,K*slot_dim], so it drops in unchanged."""
    if kind == "pool":
        return ViTEncoder(img, patch, in_ch, D_Z, depth)
    if kind == "slot":
        enc = SlotEncoder(in_channels=in_ch, vit_dim=192, patch=patch, depth=depth,
                          n_slots=N_SLOTS, slot_dim=SLOT_DIM)
        assert enc.d_z == D_Z, f"slot d_z {enc.d_z} != {D_Z}"
        return enc
    raise ValueError(kind)


def train_world_model_enc(transitions, steps, device, encoder="pool", lr=3e-4,
                          batch=128, log=print, tag=""):
    """EXACT replica of r1_imitation_fails.train_world_model's loss recipe, parameterized
    by encoder choice. Loss = pred(plain-MSE latent, stop-grad target) + 1.0*hinge
    + 5.0*(arm + tgt) with dec_arm/dec_tgt -> NORMALIZED [0,1]. Preserved verbatim so
    the slot-vs-pool comparison is clean (only the encoder differs)."""
    frames, actions, pos, tgt, idx = transitions
    H = frames.shape[-1]; adim = actions.shape[1]
    batch = min(batch, max(8, len(idx)))
    enc = build_encoder(encoder, H).to(device)
    dyn = LatentDynamics(D_Z, adim, 4).to(device)
    dec_arm = DecodeHead(D_Z, out_dim=2).to(device)                  # -> normalized fingertip [0,1]
    dec_tgt = DecodeHead(D_Z, out_dim=2).to(device)                  # -> normalized target  [0,1]
    params = (list(enc.parameters()) + list(dyn.parameters())
              + list(dec_arm.parameters()) + list(dec_tgt.parameters()))
    opt = torch.optim.AdamW(params, lr=lr)
    fr = torch.from_numpy(frames); ac = torch.from_numpy(actions)
    po = torch.from_numpy(pos); tg = torch.from_numpy(tgt); rng = np.random.RandomState(0)
    t0 = time.time()
    for step in range(steps):
        b = rng.choice(idx, batch)
        x0 = fr[b].float().to(device) / 255.0
        x1 = fr[b + 1].float().to(device) / 255.0
        a = ac[b].to(device); p0 = po[b].to(device) / H; g0 = tg[b].to(device) / H
        z_t = enc(x0)
        with torch.no_grad():
            z_next = enc(x1)                                          # stop-grad target
        z_pred = dyn(z_t, a)
        pred = F.mse_loss(z_pred, z_next)                            # plain latent prediction
        hinge = variance_hinge(z_t)                                  # per-dim std floor
        arm = F.mse_loss(dec_arm(z_t), p0)                           # SUPERVISED grounding (dominant)
        tgl = F.mse_loss(dec_tgt(z_t), g0)                           # anti-collapse
        loss = pred + 1.0 * hinge + 5.0 * (arm + tgl)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            log(f"[wm{tag} {step}/{steps}] pred={pred.item():.4f} std={z_t.std(0).mean().item():.3f} "
                f"arm_px={arm.item()**0.5*H:.1f} tgt_px={tgl.item()**0.5*H:.1f} ({time.time()-t0:.0f}s)",
                flush=True)
    return {"enc": enc.eval(), "dyn": dyn.eval(), "dec_arm": dec_arm.eval(),
            "dec_tgt": dec_tgt.eval(), "adim": adim, "img": H}


def subsample_transitions(transitions, D, seed=0):
    """Subsample to ~D transitions by keeping a random subset of consecutive-pair INDICES.
    Returns transitions with the FULL frame/action arrays but a restricted `idx` (the
    training loop only ever samples from idx, so this exactly caps the data budget)."""
    frames, actions, pos, tgt, idx = transitions
    if D is None or D >= len(idx):
        return transitions, len(idx)
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(idx, size=D, replace=False))
    return (frames, actions, pos, tgt, keep), D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--out", default="runs/lever4_ofjepa_result.json")
    ap.add_argument("--budgets", default="2000,8000,30000,full")  # D sweep; 'full' = all transitions
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--demos", type=int, default=40)             # only to define TRAIN-goal eval distribution
    ap.add_argument("--succ-target", type=float, default=0.70)   # the succ@12 bar defining D*
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wm_steps, args.eval_eps, args.ep_len, args.demos = 60, 4, 12, 6
        args.budgets = "1000,full"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    THRESH = (6.0, 12.0)
    rng = np.random.RandomState(args.seed)
    print(f"=== LEVER 4: OF-JEPA compositional-dynamics sample-efficiency | dev={dev} smoke={args.smoke} ===",
          flush=True)
    print("    CAVEAT: Reacher is SINGLE-OBJECT; a NULL result is INFORMATIVE (need multi-object env).",
          flush=True)

    # parse budgets
    budgets = []
    for b in args.budgets.split(","):
        b = b.strip()
        budgets.append(None if b == "full" else int(b))

    # eval distribution = TRAIN-region goals; reuse the same demo machinery as R1 so the
    # planner is evaluated on the same goal region the WM dynamics covers.
    print("[setup] generating TRAIN-region demos (defines eval goal distribution; not used for WM training)...",
          flush=True)
    demos, _ = gen_expert_demos(args.demos, "train", args.seed, args.image_size,
                                args.ep_len, args.action_repeat, rng)

    full = load_transitions(args.data)
    n_full = len(full[4])
    print(f"[setup] loaded {n_full} transitions from {args.data}", flush=True)

    # curve[encoder][D_label] = {succ@6, succ@12, mean_px, D_actual}
    curve = {"pool": {}, "slot": {}}
    for D in budgets:
        sub, D_act = subsample_transitions(full, D, seed=args.seed)
        label = "full" if D is None else str(D)
        for encoder in ("pool", "slot"):
            tag = f"_{encoder}_D{label}"
            print(f"\n[train] encoder={encoder} budget D={label} ({D_act} transitions)...", flush=True)
            wm = train_world_model_enc(sub, args.wm_steps, dev, encoder=encoder, tag=tag)
            r = eval_method("wm_cem", {"wm_cem": wm}, demos, "train", args.eval_eps,
                            args.seed, args.image_size, args.ep_len, args.action_repeat, dev, THRESH)
            curve[encoder][label] = {"succ6": r["succ"][6.0], "succ12": r["succ"][12.0],
                                     "mean_px": r["mean_px"], "D": D_act}
            print(f"      [{encoder:4} D={label:6}] succ@6={r['succ'][6.0]:.2f} "
                  f"succ@12={r['succ'][12.0]:.2f} mean_px={r['mean_px']:.1f}", flush=True)

    # ---- D* = smallest budget clearing succ@12 >= succ_target (None if never cleared) ----
    def d_star(enc):
        cleared = [(c["D"], lbl) for lbl, c in curve[enc].items() if c["succ12"] >= args.succ_target]
        return min(cleared)[0] if cleared else None     # smallest ACTUAL D that clears

    d_pool, d_slot = d_star("pool"), d_star("slot")
    full_pool = curve["pool"].get("full", {}).get("succ12", float("nan"))
    full_slot = curve["slot"].get("full", {}).get("succ12", float("nan"))

    # (E) sample-efficiency win: both must clear AND slot does so at <= 0.5x pool's budget
    check_E = (d_pool is not None and d_slot is not None and d_slot <= 0.5 * d_pool)
    # (C) full-D sanity: comparable at full budget
    check_C = abs(full_pool - full_slot) <= 0.15

    checks = {
        f"(E) SAMPLE-EFFICIENCY-OF-DYNAMICS: slot reaches succ@12>={args.succ_target} at "
        f"D*<=0.5x pool's (D*_slot={d_slot} <= 0.5*D*_pool={d_pool})": bool(check_E),
        "(C) FULL-D SANITY: |succ12_slot - succ12_pool| <= 0.15 at full budget "
        f"(slot={full_slot:.2f} pool={full_pool:.2f})": bool(check_C),
    }
    print("\n=== LEVER 4 PRE-REGISTERED GATE ===")
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'XX '}{k}")

    if check_E and check_C:
        verdict = ("COMPOSITIONAL SAMPLE-EFFICIENCY WIN even on single-object Reacher "
                   "(surprising; slot dynamics learn from less data) -- escalate to multi-object env to confirm")
    elif d_slot is not None and d_pool is not None and d_slot >= d_pool:
        verdict = ("NULL (INFORMATIVE): slot needs >= as much data as monolithic on single-object Reacher -- "
                   "NO compositional advantage HERE; this is EXPECTED -> the real test is a MULTI-OBJECT env "
                   "(see multi_object_env_sketch). Single-object scene cannot exercise object factorization.")
    else:
        verdict = ("INCONCLUSIVE (one/both encoders never cleared the succ@12 bar in this sweep, or only one did) "
                   "-- single-object caveat stands; multi-object env is the proper test.")
    print(f"\n  D*_pool={d_pool}  D*_slot={d_slot}  (succ@12 target {args.succ_target})")
    print(f"  full-budget succ@12: pool={full_pool:.2f} slot={full_slot:.2f}")
    print(f"  LEVER 4 VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"curve": curve, "d_star_pool": d_pool, "d_star_slot": d_slot,
               "full_succ12_pool": full_pool, "full_succ12_slot": full_slot,
               "checks": {k: bool(v) for k, v in checks.items()},
               "verdict": verdict, "succ_target": args.succ_target,
               "budgets": args.budgets, "single_object_caveat": True,
               "args": vars(args)}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


# ============================================================================
# multi_object_env_sketch -- THE PROPER TEST (stub; NOT wired/run here).
# ============================================================================
def multi_object_env_sketch():
    """The compositional claim's REAL test needs >=2 independently-moving objects so that
    "learn one object's dynamics, predict N" is exercisable. Two low-risk routes:

    (A) COMPOSITE RENDER (cheapest, no new physics): run TWO independent reacher envs with
        different seeds, render both, and ALPHA-COMPOSITE the two frames into one image
        (e.g. max/over-paint the arm+target pixels of env B onto env A). Action = concat of
        the two arms' actions [adim_A + adim_B]; the slot WM must bind the two arms to
        separate slots and predict each independently from a shared dynamics model. The
        monolithic WM must entangle both in one pooled vector. Decode-grounding targets
        become BOTH fingertips + BOTH targets (DecodeHead out_dim=4 or 8). Sample-efficiency
        gap should WIDEN with object count if the compositional bet is real.

        Pseudo:
            fA = render(envA); fB = render(envB)
            frame = np.maximum(fA, fB)                 # crude composite; or paint nonbg of B onto A
            action = np.concatenate([aA, aB])
            pos = np.concatenate([fingerA_px, fingerB_px]); tgt = concat targets
            # dec_arm out_dim=4, dec_tgt out_dim=4; cem cost = sum over both fingertips.

    (B) NATIVE MULTI-OBJECT SUITE (e.g. dm_control 'manipulator'/'stacker', or a 2-puck env):
        true contact/occlusion between objects -- the strongest test, higher build risk.

    Then RE-RUN this exact sweep (build_encoder + train_world_model_enc + the D-sweep are
    object-count-agnostic) and check whether D*_slot/D*_pool SHRINKS as #objects grows.
    That monotone shrink -- not a single-point single-object number -- is the real evidence
    for OF-JEPA compositional sample-efficiency."""
    raise NotImplementedError("sketch only -- see docstring for the multi-object test design")


if __name__ == "__main__":
    main()
