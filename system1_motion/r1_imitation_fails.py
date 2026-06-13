#!/usr/bin/env python3
"""R1 -- the regime where imitation fails: goal-shift on Reacher.

Mechanism under test: imitation needs coverage of (state x goal) in POLICY space; a world model needs
coverage of (state x action) in DYNAMICS space, which is goal-invariant. So we make the demos rich in
MOTIONS but narrow in GOALS, then test on SHIFTED goals. Imitation must extrapolate a goal->action policy
it never saw; the world model reuses dynamics it covered densely and re-plans to the new goal.

Methods (all see the same observations; the planner gets NO privileged ground-truth goal):
  demo_replay   nearest expert demo by initial frame, replayed open-loop
  bc            plain behavioral cloning: image -> action
  bc_goal       goal-conditioned BC: image + PERCEIVED target -> action   (the real baseline)
  wm_cem        JEPA world model (enc + LatentDynamics) + CEM-MPC, cost = decoded fingertip-to-target

Data:  world model trains on the rich RANDOM-exploration npz (goal-invariant dynamics, whole workspace);
       BC/BC_goal train on goal-directed EXPERT demos restricted to the TRAIN goal region.

Pre-registered gate (set BEFORE running):
  (C) in-distribution control: on TRAIN goals, bc_goal ~= wm_cem, both success > 0.8  (else confounded)
  (S) shift test: on TEST goals, wm_cem success > 0.7
  (G) EARNS-ITS-KEEP: wm_cem_success - bc_goal_success >= 0.30 on TEST goals
  falsification: if the gap < 0.10, the world model does NOT earn its keep in R1 -- report it, escalate.

Bridge to the dissociation experiment: --init-enc <substrate_*.pt> loads the GROUNDED encoder (the
condition whose TARGET probe passes), so target-decodability -> reaching shifted goals is end-to-end.
Reuses system1_motion.models (ViTEncoder, LatentDynamics, DecodeHead) -- the actual BLA world model.
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead, ema_update

EXTENT = 0.27   # arena half-extent (matches render_dataset)


# ----------------------------- env helpers (reuse render_dataset projection) -----------------------------
def make_env(seed, image_size):
    import mujoco
    from dm_control import suite
    env = suite.load("reacher", "easy", task_kwargs={"random": seed})
    renderer = mujoco.Renderer(env.physics.model.ptr, height=image_size, width=image_size)
    return env, renderer


def render(env, renderer, cam=0):
    renderer.update_scene(env.physics.data.ptr, camera=cam)
    return renderer.render()                                    # [H,W,3] uint8


def finger_world(env): return np.asarray(env.physics.named.data.geom_xpos["finger"][:2])
def target_world(env): return np.asarray(env.physics.named.data.geom_xpos["target"][:2])
def to_px(xy, sz):     return (np.asarray(xy) + EXTENT) / (2 * EXTENT) * sz
def region_of(target_xy):  return "train" if target_xy[0] >= 0.0 else "test"   # right half = train, left = test


# ----------------------------- privileged sim-shooting expert (demo generation only) -----------------------------
def expert_action(env, aspec, rng, n=48, repeat=2):
    """greedy 1-step shooting using the TRUE simulator -- privileged, for generating demos only."""
    s0 = env.physics.get_state().copy(); tgt = target_world(env)
    best_a, best_d = None, 1e9
    for _ in range(n):
        a = rng.uniform(aspec.minimum, aspec.maximum).astype(np.float32)
        env.physics.set_state(s0); env.physics.forward()
        for _ in range(repeat):
            env.physics.set_control(a); env.physics.step()
        d = float(np.linalg.norm(finger_world(env) - tgt))
        if d < best_d: best_d, best_a = d, a
    env.physics.set_state(s0); env.physics.forward()
    return best_a


def gen_expert_demos(n_demos, region, seed, image_size, ep_len, action_repeat, rng):
    """goal-directed demos whose TARGET lies in `region`. Returns list of dicts with frames/actions/target_px."""
    env, renderer = make_env(seed, image_size); aspec = env.action_spec()
    demos = []; tries = 0
    while len(demos) < n_demos and tries < n_demos * 20:
        tries += 1; env.reset()
        if region_of(target_world(env)) != region:
            continue
        tpx = to_px(target_world(env), image_size)
        frames, actions = [], []
        for _ in range(ep_len):
            frames.append(render(env, renderer).transpose(2, 0, 1).copy())     # [3,H,W]
            a = expert_action(env, aspec, rng, repeat=action_repeat)
            for _ in range(action_repeat):
                env.physics.set_control(a); env.physics.step()
            actions.append(a)
        demos.append({"frames": np.asarray(frames, np.uint8),                  # [T,3,H,W]
                      "actions": np.asarray(actions, np.float32),              # [T,adim]
                      "target_px": tpx.astype(np.float32),
                      "final_px": to_px(finger_world(env), image_size).astype(np.float32)})
    return demos, aspec


# ----------------------------- world model (JEPA: enc + dyn + decode heads), trained on exploration -----------------------------
def load_transitions(npz_path):
    d = np.load(npz_path)
    frames = d["frames"]; actions = d["actions"].astype(np.float32)
    pos = d["pos"].astype(np.float32); ep = d["ep_id"].astype(np.int64)
    tgt = d["target"].astype(np.float32) if "target" in d.files else pos * 0
    idx = np.where(ep[:-1] == ep[1:])[0]                                       # consecutive same-episode pairs
    return frames, actions, pos, tgt, idx


def train_world_model(npz_path, steps, device, d_z=384, lr=3e-4, batch=128, init_enc=None, log=print):
    frames, actions, pos, tgt, idx = load_transitions(npz_path)
    H = frames.shape[-1]; adim = actions.shape[1]
    enc = ViTEncoder(H, 8, 3, d_z, 6).to(device)
    if init_enc and os.path.exists(init_enc):
        sd = torch.load(init_enc, map_location=device); enc.load_state_dict(sd["enc"]); log(f"[wm] loaded grounded encoder {init_enc}")
    enc_tgt = ViTEncoder(H, 8, 3, d_z, 6).to(device); enc_tgt.load_state_dict(enc.state_dict())
    for p in enc_tgt.parameters(): p.requires_grad_(False)
    dyn = LatentDynamics(d_z, adim, 4).to(device)
    dec_arm = DecodeHead(d_z, out_dim=2).to(device)
    dec_tgt = DecodeHead(d_z, out_dim=2).to(device)
    params = list(enc.parameters()) + list(dyn.parameters()) + list(dec_arm.parameters()) + list(dec_tgt.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    fr = torch.from_numpy(frames); ac = torch.from_numpy(actions)
    po = torch.from_numpy(pos); tg = torch.from_numpy(tgt); rng = np.random.RandomState(0)
    t0 = time.time()
    for step in range(steps):
        b = rng.choice(idx, batch)
        x0 = fr[b].float().to(device) / 255.0
        x1 = fr[b + 1].float().to(device) / 255.0
        a = ac[b].to(device); p0 = po[b].to(device); g0 = tg[b].to(device)
        z0 = enc(x0)
        with torch.no_grad(): z1_t = enc_tgt(x1)
        z1_pred = dyn(z0, a)
        pred = F.mse_loss(z1_pred, z1_t)                                       # latent prediction (the world model)
        var = torch.relu(1.0 - z0.std(0)).mean()                              # variance hinge (anti-collapse)
        arm = F.mse_loss(dec_arm(z0), p0)                                     # ground fingertip (decodable)
        tgl = F.mse_loss(dec_tgt(z0), g0)                                     # ground TARGET (decodable -> plannable)
        loss = pred + 1.0 * var + 0.05 * arm + 0.05 * tgl
        opt.zero_grad(); loss.backward(); opt.step(); ema_update(enc_tgt, enc, 0.996)
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            log(f"[wm step {step}/{steps}] pred={pred.item():.3f} var={z0.std(0).mean().item():.3f} "
                f"arm_px={arm.item()**0.5:.1f} tgt_px={tgl.item()**0.5:.1f} ({time.time()-t0:.0f}s)")
    return {"enc": enc.eval(), "dyn": dyn.eval(), "dec_arm": dec_arm.eval(), "dec_tgt": dec_tgt.eval(),
            "adim": adim, "img": H}


# ----------------------------- CEM-MPC planner (uses the LEARNED model only) -----------------------------
@torch.no_grad()
def cem_plan(wm, z0, target_px, aspec, device, horizon=5, iters=3, pop=128, elite=16):
    adim = wm["adim"]; lo = torch.tensor(aspec.minimum, device=device); hi = torch.tensor(aspec.maximum, device=device)
    mu = torch.zeros(horizon, adim, device=device); sigma = torch.ones(horizon, adim, device=device) * 0.5
    tpx = torch.tensor(target_px, device=device).float()
    for _ in range(iters):
        seqs = (mu[None] + sigma[None] * torch.randn(pop, horizon, adim, device=device)).clamp(lo, hi)
        z = z0.expand(pop, -1).clone()
        cost = torch.zeros(pop, device=device)
        for h in range(horizon):
            z = wm["dyn"](z, seqs[:, h])
            d = (wm["dec_arm"](z) - tpx[None]).pow(2).sum(-1).sqrt()           # predicted fingertip-to-target px
            cost = cost + d
        elite_idx = cost.topk(elite, largest=False).indices
        e = seqs[elite_idx]; mu = e.mean(0); sigma = e.std(0) + 1e-3
    return mu[0].cpu().numpy()                                                # first action (MPC)


# ----------------------------- behavioral cloning baselines -----------------------------
class BCNet(nn.Module):
    def __init__(self, img, adim, goal_cond):
        super().__init__()
        self.enc = ViTEncoder(img, 8, 3, 256, 4)
        self.goal_cond = goal_cond
        self.head = nn.Sequential(nn.Linear(256 + (2 if goal_cond else 0), 256), nn.ReLU(), nn.Linear(256, adim))
    def forward(self, x, g=None):
        z = self.enc(x)
        if self.goal_cond: z = torch.cat([z, g], -1)
        return self.head(z)


def train_bc(demos, img, adim, device, steps, goal_cond, lr=3e-4, batch=128, log=print):
    X = np.concatenate([d["frames"][:-0 or None] for d in demos], 0).astype(np.float32) / 255.0
    A = np.concatenate([d["actions"] for d in demos], 0).astype(np.float32)
    G = np.concatenate([np.tile(d["target_px"], (len(d["actions"]), 1)) for d in demos], 0).astype(np.float32)
    X, A, G = X[:len(A)], A, G                                                # align lengths
    net = BCNet(img, adim, goal_cond).to(device); opt = torch.optim.AdamW(net.parameters(), lr=lr)
    Xt = torch.from_numpy(X); At = torch.from_numpy(A); Gt = torch.from_numpy(G); rng = np.random.RandomState(0)
    for step in range(steps):
        b = rng.choice(len(A), batch)
        x = Xt[b].to(device); a = At[b].to(device); g = Gt[b].to(device) / img
        pred = net(x, g if goal_cond else None)
        loss = F.mse_loss(pred, a)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            log(f"[bc{'_goal' if goal_cond else ''} step {step}/{steps}] loss={loss.item():.4f}")
    return net.eval()


# ----------------------------- evaluation in the live env -----------------------------
@torch.no_grad()
def eval_method(method, wm, bc, bc_goal, demos, region, n_eps, seed0, image_size, ep_len, action_repeat, device, thresh_px):
    succ, dists = 0, []
    for e in range(n_eps):
        env, renderer = make_env(seed0 + 1000 + e, image_size); env.reset()
        # reset until target in eval region
        for _ in range(200):
            if region_of(target_world(env)) == region: break
            env.reset()
        aspec = env.action_spec(); tpx = to_px(target_world(env), image_size)
        if method == "demo_replay":
            x0 = render(env, renderer).transpose(2, 0, 1).astype(np.float32) / 255.0
            dd = [np.mean((d["frames"][0].astype(np.float32) / 255.0 - x0) ** 2) for d in demos]
            acts = demos[int(np.argmin(dd))]["actions"]
            for t in range(ep_len):
                a = acts[min(t, len(acts) - 1)]
                for _ in range(action_repeat): env.physics.set_control(a); env.physics.step()
        else:
            for t in range(ep_len):
                x = torch.from_numpy(render(env, renderer).transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(device)
                if method == "wm_cem":
                    z0 = wm["enc"](x)
                    a = cem_plan(wm, z0, tpx, aspec, device)
                elif method == "bc":
                    a = bc(x).cpu().numpy()[0]
                elif method == "bc_goal":
                    g = torch.tensor(tpx / image_size, device=device).float()[None]
                    a = bc_goal(x, g).cpu().numpy()[0]
                a = np.clip(a, aspec.minimum, aspec.maximum)
                for _ in range(action_repeat): env.physics.set_control(a); env.physics.step()
        d = float(np.linalg.norm(to_px(finger_world(env), image_size) - tpx))
        dists.append(d); succ += int(d < thresh_px)
    return {"success": succ / n_eps, "mean_px": float(np.mean(dists))}


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--init-enc", default="")
    ap.add_argument("--out", default="runs/r1_result.json")
    ap.add_argument("--wm-steps", type=int, default=8000)
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--demos", type=int, default=120)
    ap.add_argument("--eval-eps", type=int, default=40)
    ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--threshold-px", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wm_steps, args.bc_steps, args.demos, args.eval_eps, args.ep_len = 80, 60, 6, 4, 12
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)
    print(f"=== R1: the regime where imitation fails (goal-shift Reacher) | device={dev} smoke={args.smoke} ===", flush=True)

    print("[1/4] generating narrow-goal (TRAIN region) expert demos...", flush=True)
    demos, _ = gen_expert_demos(args.demos, "train", args.seed, args.image_size, args.ep_len, args.action_repeat, rng)
    print(f"      {len(demos)} expert demos; mean final finger-target = "
          f"{np.mean([np.linalg.norm(d['final_px']-d['target_px']) for d in demos]):.1f}px", flush=True)

    print("[2/4] training the JEPA world model on rich exploration (goal-invariant dynamics)...", flush=True)
    wm = train_world_model(args.data, args.wm_steps, dev, init_enc=args.init_enc or None)
    img, adim = wm["img"], wm["adim"]

    print("[3/4] training BC and goal-conditioned BC on the narrow-goal demos...", flush=True)
    bc = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=False)
    bc_goal = train_bc(demos, img, adim, dev, args.bc_steps, goal_cond=True)

    print("[4/4] evaluating all methods on TRAIN goals (in-dist control) and TEST goals (shift)...", flush=True)
    R = {}
    for region in ("train", "test"):
        R[region] = {}
        for m in ("demo_replay", "bc", "bc_goal", "wm_cem"):
            R[region][m] = eval_method(m, wm, bc, bc_goal, demos, region, args.eval_eps, args.seed,
                                       args.image_size, args.ep_len, args.action_repeat, dev, args.threshold_px)
            print(f"      [{region:5}] {m:11} success={R[region][m]['success']:.2f}  mean_px={R[region][m]['mean_px']:.1f}", flush=True)

    wm_s, bcg_s = R["test"]["wm_cem"]["success"], R["test"]["bc_goal"]["success"]
    gap = wm_s - bcg_s
    checks = {
        "(C) in-dist control: bc_goal ~= wm_cem on TRAIN goals, both > 0.8 (else confounded)":
            R["train"]["bc_goal"]["success"] > 0.8 and R["train"]["wm_cem"]["success"] > 0.8,
        "(S) shift: wm_cem success > 0.7 on TEST goals": wm_s > 0.7,
        "(G) EARNS ITS KEEP: wm_cem - bc_goal >= 0.30 on TEST goals": gap >= 0.30,
    }
    print("\n=== R1 PRE-REGISTERED GATE ===")
    for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
    verdict = "WORLD MODEL EARNS ITS KEEP" if all(checks.values()) else \
              ("FALSIFIED (world model does NOT beat imitation here -- escalate the ladder)" if gap < 0.10 else "INCONCLUSIVE")
    print(f"\n  shift-region gap (wm_cem - bc_goal) = {gap:+.2f}")
    print(f"  R1 VERDICT: {verdict}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"results": R, "gap": gap, "checks": {k: bool(v) for k, v in checks.items()}, "verdict": verdict,
               "args": vars(args)}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
