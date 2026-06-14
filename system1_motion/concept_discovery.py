#!/usr/bin/env python3
"""ABSTRACTION-BARRIER probe: does a world-model latent DISCOVER structure that
was NOT in its supervision, purely from prediction pressure?

==========================================================================
PRE-REGISTERED EXPECTATION: NULL is the EXPECTED, INFORMATIVE outcome here.
==========================================================================
This is a WEAK test of the Abstraction Barrier, on purpose, because Reacher is a
MINIMAL, single-object, 2-variable scene (controllable fingertip + uncontrolled
target). The prior grounding-DISSOCIATION result (system1_motion/train_dissoc.py)
already showed the relevant mechanism on exactly this env: action/inverse-dynamics
grounding rescued ONLY the controllable variable, and ONLY directly decode-
supervised variables got represented -- target-grounding -> target decodable
~0.1px while everything un-supervised stayed at ~18px (chance / mean-prediction
level). By the same logic, a world model that grounds ONLY the fingertip and is
NEVER supervised on the target has no decode pressure to encode the target, and a
single mean-pooled latent on a one-object scene has weak prediction pressure to do
so either. So we EXPECT the arm-only WM's target probe to land near random-encoder
level (NO emergence = NULL).

A NULL here is INFORMATIVE, not a failure: it says concept discovery (latent
structure beyond supervision) needs a RICHER / MULTI-OBJECT environment with real
prediction pressure to represent un-supervised entities. It mirrors the OF-JEPA
single-object caveat (identity binding needs multiple objects to bite). The REAL
test of the Abstraction Barrier is a multi-object / compositional env -- see the
"honest risks" in the run report. This script PINS DOWN the single-object baseline
so the multi-object result has a measured control to beat.

WHAT WE ACTUALLY MEASURE
------------------------
Three world models, then a FRESH held-out probe of the TARGET (px) from the FROZEN
encoder latent enc(frame) -- the dissociation probe methodology:

  WM_arm   grounds ONLY the fingertip (arm). Target is NEVER supervised.
           -> target probe tests UNSUPERVISED EMERGENCE (the headline).
  WM_both  grounds fingertip AND target (control / sanity).
           -> target probe should PASS (<=5px): proves target IS decodable when
              the right channel grounds it (rules out "not in the pixels").
  RANDOM   untrained encoder, no training at all (chance / mean-prediction).
           -> target probe ~ workspace radius in px: the NULL reference level.

PRE-REGISTERED GATE (set BEFORE running):
  (C) SANITY      : WM_both target probe <= 5px           (the probe works at all)
  (D) DISCOVERY   : WM_arm  target probe <  8px           (target EMERGED unsupervised)
  NULL (expected) : WM_arm  target probe >= ~15px (~random-encoder level) => NO emergence

The arm-only WM keeps the SAME audit-hardened convergence gate as the source
trainer (arm_px <= arm_gate_px, early-abort + reinit) so the WM is verified-ALIVE
before we trust its (negative) probe -- a dead WM must never produce a fake null.

Reuses system1_motion/r1_imitation_fails.py (load_transitions, EXTENT) and
system1_motion/models.py (ViTEncoder, LatentDynamics, DecodeHead). The arm-only
training loop is a thin LOCAL copy of train_world_model with the target-decode term
removed (5x*tgl -> 0) and EVERYTHING ELSE preserved exactly.

    python -m system1_motion.concept_discovery --data runs/reacher_transitions.npz
    python -m system1_motion.concept_discovery --smoke
"""
from __future__ import annotations

import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead
from system1_motion.objective import variance_hinge
# Reuse the verified data loader + the canonical FULL (arm+target) trainer + audit rollout.
from system1_motion.r1_imitation_fails import (
    load_transitions, train_world_model, rollout_error_px, EXTENT)


# ----------------------------------------------------------------------------
# (1) ARM-ONLY world model: thin local copy of train_world_model's loop with the
#     TARGET grounding term removed. Everything else is preserved EXACTLY:
#       - plain-MSE latent prediction with stop-grad target encoder
#       - 15x arm (fingertip) decode grounding
#       - 1.0x variance_hinge anti-collapse floor
#       - per-attempt torch+numpy seeding (F3), fixed data order across attempts
#       - EARLY-ABORT at 45% if arm_px > early_px -> reinit, up to max_attempts
#       - CONVERGENCE GATE: only return a WM whose final arm_px <= arm_gate_px
#       - held-out (in-dist) + optional OOD rollout audit; raise if never converged
#     ONLY DIFFERENCE vs train_world_model: the `5.0 * tgl` target term is dropped
#     from the loss AND dec_tgt is not optimized (target is genuinely UNSUPERVISED).
# ----------------------------------------------------------------------------
def train_world_model_arm_only(transitions, steps, device, d_z=384, lr=3e-4, batch=128,
                               beta_var=1.0, init_enc=None, log=print, tag="_arm", seed=0,
                               arm_gate_px=5.0, early_px=9.0, max_attempts=6, rollout_eval=None):
    """JEPA WM grounding ONLY the fingertip (arm). Target is NEVER supervised, so any
    target-decodability of the frozen latent is EMERGENT (the abstraction-barrier test).

    Loss: plain-MSE latent prediction (stop-grad target) + 15x arm + var floor.
    (vs train_world_model: the 5x target-decode grounding term is REMOVED.)"""
    frames, actions, pos, tgt, idx = transitions
    H = frames.shape[-1]; adim = actions.shape[1]
    batch = min(batch, max(8, len(idx)))
    fr = torch.from_numpy(frames); ac = torch.from_numpy(actions)
    po = torch.from_numpy(pos)                                   # tgt is intentionally NOT used in training
    # Same held-out split as the source trainer (never measure rollout on trained-on transitions).
    split = int(0.9 * len(idx)); train_idx = idx[:split]
    held_idx = idx[split:] if (len(idx) - split) >= 64 else idx
    held_trans = (frames, actions, pos, tgt, held_idx)
    early_step = int(0.45 * steps)
    last = None
    for attempt in range(max_attempts):
        torch.manual_seed(seed + 1000 * attempt); np.random.seed(seed + attempt)        # F3
        enc = ViTEncoder(H, 8, 3, d_z, 6).to(device)
        if init_enc and os.path.exists(init_enc):
            sd = torch.load(init_enc, map_location=device); enc.load_state_dict(sd["enc"])
            log(f"[wm{tag}] loaded grounded encoder {init_enc}")
        dyn = LatentDynamics(d_z, adim, 4).to(device)
        dec_arm = DecodeHead(d_z, out_dim=2).to(device)                 # ONLY the arm head is optimized
        opt = torch.optim.AdamW(list(enc.parameters()) + list(dyn.parameters())
                                + list(dec_arm.parameters()), lr=lr)
        brng = np.random.RandomState(0)                                 # fixed data order -> init is the only variable
        t0 = time.time(); cur_arm = 99.0; stuck = False
        for step in range(steps):
            b = brng.choice(train_idx, batch)
            x0 = fr[b].float().to(device) / 255.0; x1 = fr[b + 1].float().to(device) / 255.0
            a = ac[b].to(device); p0 = po[b].to(device) / H
            z_t = enc(x0)
            with torch.no_grad(): z_next = enc(x1)                      # stop-grad target
            pred = F.mse_loss(dyn(z_t, a), z_next)
            hinge = variance_hinge(z_t)
            arm = F.mse_loss(dec_arm(z_t), p0)
            loss = pred + 1.0 * hinge + 15.0 * arm                      # <-- NO target term (5x*tgl removed)
            opt.zero_grad(); loss.backward(); opt.step()
            if step == early_step or step % max(1, steps // 10) == 0 or step == steps - 1:
                cur_arm = arm.item() ** 0.5 * H
                log(f"[wm{tag} a{attempt+1} step {step}/{steps}] pred={pred.item():.4f} "
                    f"std={z_t.std(0).mean().item():.3f} arm_px={cur_arm:.1f} ({time.time()-t0:.0f}s)", flush=True)
            if step == early_step and cur_arm > early_px:
                log(f"[wm{tag} a{attempt+1}] EARLY-ABORT arm_px={cur_arm:.1f}>{early_px} at {step} -> reinit",
                    flush=True); stuck = True; break
        if stuck: continue
        roll = rollout_error_px(enc.eval(), dyn.eval(), dec_arm.eval(), held_trans, device)
        roll_ood = (rollout_error_px(enc.eval(), dyn.eval(), dec_arm.eval(), rollout_eval, device)
                    if rollout_eval is not None else float("nan"))
        wm = {"enc": enc.eval(), "dyn": dyn.eval(), "dec_arm": dec_arm.eval(),
              "adim": adim, "img": H, "arm_px": cur_arm, "rollout_px": roll,
              "rollout_ood_px": roll_ood, "attempts": attempt + 1}
        if cur_arm <= arm_gate_px:
            log(f"[wm{tag}] CONVERGED arm_px={cur_arm:.1f}<={arm_gate_px} rollout_heldout={roll:.1f} "
                f"(attempt {attempt+1})", flush=True)
            wm["converged"] = True; return wm
        log(f"[wm{tag} a{attempt+1}] GATE FAIL arm_px={cur_arm:.1f}>{arm_gate_px}; retry", flush=True); last = wm
    raise RuntimeError(f"[wm{tag}] arm-only WM did NOT converge in {max_attempts} attempts "
                       f"(best arm_px={last['arm_px']:.1f} > {arm_gate_px}px). Refusing to return an "
                       f"unverified world model (audit N2).")


# ----------------------------------------------------------------------------
# (2) FRESH probe of the TARGET from the FROZEN encoder latent (held-out split).
#     Mirrors the dissociation probe: a brand-new MLP readout trained on enc(frame)
#     -> target px, evaluated on a disjoint split, reported in pixels. The encoder
#     is frozen; only the probe learns. If the target is not linearly/MLP-decodable
#     from the latent, this lands near the mean-prediction baseline (the NULL).
# ----------------------------------------------------------------------------
@torch.no_grad()
def _embed(enc, frames_np, idx, device, bs=512):
    enc.eval(); Z = []
    fr = torch.from_numpy(frames_np)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        Z.append(enc(fr[b].float().to(device) / 255.0).cpu())
    return torch.cat(Z, 0)


def probe_px(enc, frames_np, target_px, idx, device, steps=4000, hidden=256, seed=0, log=print, tag=""):
    """Train a FRESH MLP probe latent->target_px on 80% of `idx`, eval px error on 20%.
    Returns dict with held-out mean L2 px error and the mean-prediction baseline px."""
    torch.manual_seed(seed); np.random.seed(seed)
    Z = _embed(enc, frames_np, idx, device)                            # [N, d_z] on cpu
    Y = torch.from_numpy(target_px[idx].astype(np.float32))            # [N, 2] px
    n = len(idx); perm = np.random.RandomState(seed).permutation(n)
    ntr = int(0.8 * n); tr = perm[:ntr]; te = perm[ntr:]
    Ztr, Ytr = Z[tr].to(device), Y[tr].to(device)
    Zte, Yte = Z[te].to(device), Y[te].to(device)
    # mean-prediction baseline (predict train-mean target for everyone): the chance level.
    mean_y = Ytr.mean(0, keepdim=True)
    base_px = (Yte - mean_y).pow(2).sum(-1).sqrt().mean().item()
    net = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device=device).manual_seed(seed)
    for step in range(steps):
        s = torch.randint(0, len(tr), (256,), generator=g, device=device)
        pred = net(Ztr[s])
        loss = F.mse_loss(pred, Ytr[s])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            with torch.no_grad():
                te_px = (net(Zte) - Yte).pow(2).sum(-1).sqrt().mean().item()
            log(f"[probe{tag} step {step}/{steps}] train_mse={loss.item():.4f} heldout_px={te_px:.1f}", flush=True)
    with torch.no_grad():
        te_px = (net(Zte) - Yte).pow(2).sum(-1).sqrt().mean().item()
    return {"target_probe_px": round(te_px, 2), "mean_baseline_px": round(base_px, 2),
            "n_train": int(ntr), "n_test": int(n - ntr)}


# ----------------------------------------------------------------------------
# (3) main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--out", default="runs/concept_discovery_result.json")
    ap.add_argument("--init-enc", default="")
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--probe-steps", type=int, default=4000)
    ap.add_argument("--discovery-px", type=float, default=8.0)         # (D) target<this => emergence
    ap.add_argument("--sanity-px", type=float, default=5.0)            # (C) control target<=this => probe works
    ap.add_argument("--null-px", type=float, default=15.0)             # >= this ~ random-encoder => NULL
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wm_steps, args.probe_steps = 80, 200
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CONCEPT-DISCOVERY (abstraction-barrier) | device={dev} smoke={args.smoke} ===", flush=True)
    print("    REMINDER: NULL (no emergence) is the EXPECTED, INFORMATIVE outcome on "
          "single-object Reacher.\n    The real test of the Abstraction Barrier is a "
          "multi-object/richer env -- this PINS the baseline.", flush=True)

    transitions = load_transitions(args.data)
    frames, actions, pos, tgt, idx = transitions
    img = frames.shape[-1]
    print(f"[data] frames={frames.shape} transitions={len(idx)} img={img}px", flush=True)

    # smoke vs full convergence-gate settings (smoke loosens the gate so it runs anywhere).
    if args.smoke:
        gate = dict(arm_gate_px=99.0, early_px=99.0, max_attempts=1)
    else:
        gate = dict(arm_gate_px=5.0, early_px=9.0, max_attempts=6)

    # ---- WM_both: control that DOES ground the target (reuse the canonical trainer) ----
    print("\n[1/3] WM_both -- grounds fingertip AND target (CONTROL: target SHOULD be decodable)...", flush=True)
    wm_both = train_world_model(transitions, args.wm_steps, dev, init_enc=args.init_enc or None,
                                tag="_both", seed=args.seed, **gate)

    # ---- WM_arm: grounds ONLY the fingertip; target NEVER supervised (the headline test) ----
    print("\n[2/3] WM_arm -- grounds ONLY fingertip; target NEVER supervised (EMERGENCE test)...", flush=True)
    wm_arm = train_world_model_arm_only(transitions, args.wm_steps, dev, init_enc=args.init_enc or None,
                                        tag="_arm", seed=args.seed, **gate)

    # ---- RANDOM: untrained encoder = chance / mean-prediction reference level ----
    print("\n[3/3] RANDOM -- untrained encoder (chance / mean-prediction reference)...", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    enc_rand = ViTEncoder(img, 8, 3, 384, 6).to(dev).eval()

    # ---- FRESH target probes on each frozen encoder (held-out split) ----
    print("\n[probe] fresh held-out TARGET probes (target px from frozen enc(frame))...", flush=True)
    pr_both = probe_px(wm_both["enc"], frames, tgt, idx, dev, args.probe_steps, seed=args.seed, tag="_both")
    pr_arm  = probe_px(wm_arm["enc"],  frames, tgt, idx, dev, args.probe_steps, seed=args.seed, tag="_arm")
    pr_rand = probe_px(enc_rand,       frames, tgt, idx, dev, args.probe_steps, seed=args.seed, tag="_rand")
    print(f"      WM_both target probe = {pr_both['target_probe_px']}px (mean-baseline {pr_both['mean_baseline_px']}px)", flush=True)
    print(f"      WM_arm  target probe = {pr_arm['target_probe_px']}px (mean-baseline {pr_arm['mean_baseline_px']}px)", flush=True)
    print(f"      RANDOM  target probe = {pr_rand['target_probe_px']}px (mean-baseline {pr_rand['mean_baseline_px']}px)", flush=True)

    # ---- PRE-REGISTERED GATE ----
    both_px = pr_both["target_probe_px"]; arm_px = pr_arm["target_probe_px"]; rand_px = pr_rand["target_probe_px"]
    checks = {
        "(C) SANITY: target-grounded control WM target probe <= %.0fpx (probe works)" % args.sanity_px:
            both_px <= args.sanity_px,
        "(D) DISCOVERY: arm-only WM target probe < %.0fpx (target EMERGED unsupervised)" % args.discovery_px:
            arm_px < args.discovery_px,
    }
    is_null = arm_px >= args.null_px                  # >= ~random-encoder level => NO emergence
    print("\n=== PRE-REGISTERED GATE ===")
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")

    sanity_ok = checks[list(checks)[0]]
    discovery = checks[list(checks)[1]]
    if not sanity_ok:
        verdict = ("INCONCLUSIVE -- the CONTROL probe failed (target not decodable even when grounded). "
                   "Probe/data broken; cannot interpret the arm-only result.")
    elif discovery:
        verdict = ("DISCOVERY -- target EMERGED in the arm-only latent UNSUPERVISED (%.1fpx < %.0fpx). "
                   "Prediction pressure alone crossed the abstraction barrier on single-object Reacher "
                   "(SURPRISING vs the dissociation prior -- scrutinize before believing)."
                   % (arm_px, args.discovery_px))
    elif is_null:
        verdict = ("NULL (EXPECTED, INFORMATIVE) -- NO unsupervised emergence: arm-only target probe "
                   "%.1fpx >= ~random-encoder level %.1fpx. On a single-object 2-variable scene there is "
                   "no decode pressure and weak prediction pressure to represent the un-supervised target. "
                   "Consistent with the grounding-dissociation finding. The REAL abstraction-barrier test "
                   "needs a MULTI-OBJECT / richer env -- this run pins the single-object baseline to beat."
                   % (arm_px, rand_px))
    else:
        verdict = ("PARTIAL -- arm-only target probe %.1fpx is between the discovery (%.0fpx) and null "
                   "(%.0fpx) thresholds: weak/ambiguous emergence. Treat as effectively NULL on this "
                   "single-object env; defer to the multi-object test." % (arm_px, args.discovery_px, args.null_px))

    print(f"\n  target probe px: WM_both(control)={both_px} | WM_arm(test)={arm_px} | RANDOM(chance)={rand_px}")
    print(f"  VERDICT: {verdict}")

    out = {
        "verdict": verdict,
        "is_null_expected_informative": bool(is_null),
        "checks": {k: bool(v) for k, v in checks.items()},
        "thresholds_px": {"sanity": args.sanity_px, "discovery": args.discovery_px, "null": args.null_px},
        "target_probe_px": {"wm_both_control": both_px, "wm_arm_test": arm_px, "random_chance": rand_px},
        "probe_detail": {"wm_both": pr_both, "wm_arm": pr_arm, "random": pr_rand},
        "wm_arm": {"arm_px": wm_arm["arm_px"], "rollout_px": wm_arm["rollout_px"],
                   "attempts": wm_arm["attempts"], "converged": wm_arm.get("converged", False)},
        "wm_both": {"arm_px": wm_both["arm_px"], "rollout_px": wm_both["rollout_px"],
                    "attempts": wm_both["attempts"], "converged": wm_both.get("converged", False)},
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
