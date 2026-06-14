#!/usr/bin/env python3
"""Backend for the interactive "drag-the-goal" world-model demo.

Steps 1-2 of the demo build:
  (1) checkpoint save/load for a GATED world model + a goal-conditioned BC.
  (2) a headless DemoEngine: hold a live Reacher env, let a caller SET A GOAL anywhere (incl. the shifted
      region the demos never covered), STEP either method (world-model CEM planner or imitation BC), and --
      for the world model -- return the planner's IMAGINED rollout (decoded fingertip trajectory) for the
      "imagination" overlay. The web GUI is a rendering shell on top of this.

CLI:
  --train      : train a gated WM (exploration) + goal-conditioned BC (train-region demos), save checkpoints.
  --selfcheck  : load checkpoints, drag a goal into the SHIFTED region, run WM vs BC for an episode, and
                 confirm the WM reaches it while BC fails -- the demo's money shot, headless.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch

from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead
from system1_motion.r1_imitation_fails import (
    make_env, render, finger_world, target_world, to_px, region_of, EXTENT,
    gen_expert_demos, load_transitions, train_world_model, train_bc, BCNet)

CKPT = "runs/demo_ckpt"


# ----------------------------- checkpoint save/load -----------------------------
def save_wm(wm, path):
    torch.save({"enc": wm["enc"].state_dict(), "dyn": wm["dyn"].state_dict(),
                "dec_arm": wm["dec_arm"].state_dict(), "dec_tgt": wm["dec_tgt"].state_dict(),
                "d_z": 384, "adim": wm["adim"], "img": wm["img"],
                "arm_px": wm.get("arm_px"), "rollout_ood_px": wm.get("rollout_ood_px")}, path)


def load_wm(path, device):
    ck = torch.load(path, map_location=device); H, adim, d_z = ck["img"], ck["adim"], ck["d_z"]
    enc = ViTEncoder(H, 8, 3, d_z, 6).to(device); enc.load_state_dict(ck["enc"]); enc.eval()
    dyn = LatentDynamics(d_z, adim, 4).to(device); dyn.load_state_dict(ck["dyn"]); dyn.eval()
    da = DecodeHead(d_z, out_dim=2).to(device); da.load_state_dict(ck["dec_arm"]); da.eval()
    dt = DecodeHead(d_z, out_dim=2).to(device); dt.load_state_dict(ck["dec_tgt"]); dt.eval()
    return {"enc": enc, "dyn": dyn, "dec_arm": da, "dec_tgt": dt, "adim": adim, "img": H,
            "arm_px": ck.get("arm_px"), "rollout_ood_px": ck.get("rollout_ood_px")}


def save_bc(net, path, img, adim):
    torch.save({"state": net.state_dict(), "img": img, "adim": adim}, path)


def load_bc(path, device):
    ck = torch.load(path, map_location=device)
    net = BCNet(ck["img"], ck["adim"], goal_cond=True).to(device); net.load_state_dict(ck["state"]); net.eval()
    return net


# ----------------------------- CEM planner + imagination -----------------------------
@torch.no_grad()
def cem_imagine(wm, z0, goal_px_norm, aspec, device, horizon=6, iters=4, pop=160, elite=20, terminal_w=4.0):
    """returns (first_action, imagined_fingertip_px[horizon,2]) -- the planner's chosen future, for overlay."""
    adim = wm["adim"]
    lo = torch.tensor(aspec.minimum, device=device, dtype=torch.float32)
    hi = torch.tensor(aspec.maximum, device=device, dtype=torch.float32)
    mu = torch.zeros(horizon, adim, device=device); sigma = torch.ones(horizon, adim, device=device) * 0.5
    g = torch.tensor(goal_px_norm, device=device).float()
    for _ in range(iters):
        seqs = (mu[None] + sigma[None] * torch.randn(pop, horizon, adim, device=device)).clamp(lo, hi)
        z = z0.expand(pop, -1).clone(); cost = torch.zeros(pop, device=device)
        for h in range(horizon):
            z = wm["dyn"](z, seqs[:, h])
            d = (wm["dec_arm"](z) - g[None]).pow(2).sum(-1).sqrt()
            cost = cost + d * (terminal_w if h == horizon - 1 else 1.0)
        e = seqs[cost.topk(elite, largest=False).indices]; mu = e.mean(0); sigma = e.std(0) + 1e-3
    # roll the chosen plan, decode fingertip px at each step = the imagined trajectory
    z = z0.clone(); imagined = []
    for h in range(horizon):
        z = wm["dyn"](z, mu[h:h + 1])
        imagined.append((wm["dec_arm"](z)[0].cpu().numpy() * wm["img"]).tolist())
    return mu[0].cpu().numpy(), imagined


# ----------------------------- the live demo engine -----------------------------
class DemoEngine:
    def __init__(self, wm, bc, device, image_size=64, seed=0, action_repeat=2, disp_size=440):
        import mujoco
        self.wm, self.bc, self.dev = wm, bc, device
        self.img, self.ar, self.disp_size = image_size, action_repeat, disp_size
        self.env, self.renderer = make_env(seed, image_size); self.aspec = self.env.action_spec()
        self.disp = mujoco.Renderer(self.env.physics.model.ptr, height=disp_size, width=disp_size)
        self.trail = []                                          # fingertip history (display px) for motion trail
        self.reset()

    def reset(self):
        self.env.reset(); self.trail = []; return self._obs()

    def set_goal(self, xy_world):
        """move the Reacher target to the user's dragged goal. dm_control Reacher positions the target via
        model.geom_pos['target'] (a static geom), NOT a joint -- setting qpos does nothing."""
        xy = np.clip(np.asarray(xy_world, np.float64), -EXTENT * 0.8, EXTENT * 0.8)
        gp = self.env.physics.named.model.geom_pos
        gp["target", "x"] = xy[0]; gp["target", "y"] = xy[1]
        self.env.physics.forward()
        return {"goal_world": xy.tolist(), "goal_px": to_px(xy, self.img).tolist()}

    @torch.no_grad()
    def step(self, method):
        x = torch.from_numpy(render(self.env, self.renderer).transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(self.dev)
        tpx = to_px(target_world(self.env), self.img); imagined = None
        z0 = self.wm["enc"](x)
        if method == "wm":
            cur_px = float(np.linalg.norm(self.wm["dec_arm"](z0)[0].cpu().numpy() * self.img - tpx))
            if cur_px < 14.0:                                   # NEAR goal: precision mode -- deep search, settle on target
                a, imagined = cem_imagine(self.wm, z0, tpx / self.img, self.aspec, self.dev,
                                          horizon=8, iters=8, pop=384, elite=32, terminal_w=10.0)
            else:                                               # FAR: fast mode -- light search, quick approach (keeps fps up)
                a, imagined = cem_imagine(self.wm, z0, tpx / self.img, self.aspec, self.dev,
                                          horizon=6, iters=4, pop=160, elite=20, terminal_w=4.0)
        elif method == "bc":
            g = torch.tensor(tpx / self.img, device=self.dev).float()[None]; a = self.bc(x, g).cpu().numpy()[0]
        else:
            a = np.zeros(self.aspec.shape[0], np.float32)
        a = np.clip(a, self.aspec.minimum, self.aspec.maximum)
        for _ in range(self.ar): self.env.physics.set_control(a); self.env.physics.step()
        return self._obs(imagined)

    def _obs(self, imagined=None):
        fr = render(self.env, self.disp)                        # HIGH-RES display frame (model still sees 64px)
        f_w, t_w = finger_world(self.env), target_world(self.env)
        sc, k = self.disp_size, self.disp_size / self.img
        fpx, tpx = to_px(f_w, sc), to_px(t_w, sc)               # display-space pixel coords
        self.trail.append(fpx.tolist()); self.trail = self.trail[-20:]
        dist64 = float(np.linalg.norm(to_px(f_w, self.img) - to_px(t_w, self.img)))  # metric in MODEL px (64)
        imag = [[p[0] * k, p[1] * k] for p in imagined] if imagined is not None else None
        return {"frame": fr, "finger_px": fpx.tolist(), "target_px": tpx.tolist(), "trail": list(self.trail),
                "dist_px": dist64, "disp": sc, "imagined_px": imag, "region": region_of(t_w),
                "wm_arm_px": self.wm.get("arm_px"), "wm_rollout_ood_px": self.wm.get("rollout_ood_px")}


# ----------------------------- train + save + selfcheck -----------------------------
def train_and_save(args, device):
    os.makedirs(CKPT, exist_ok=True)
    print("[demo] training gated WM on exploration...", flush=True)
    expl = load_transitions(args.data)
    wm = train_world_model(expl, args.wm_steps, device, tag="_demo", rollout_eval=expl)
    save_wm(wm, f"{CKPT}/wm.pt"); print(f"[demo] saved gated WM (arm_px={wm['arm_px']:.1f}, OOD={wm['rollout_ood_px']:.1f})", flush=True)
    print("[demo] training goal-conditioned BC on TRAIN-region demos...", flush=True)
    demos, _ = gen_expert_demos(args.demos, "train", args.seed, args.image_size, args.ep_len, args.action_repeat, np.random.RandomState(args.seed))
    bc = train_bc(demos, wm["img"], wm["adim"], device, args.bc_steps, goal_cond=True)
    save_bc(bc, f"{CKPT}/bc.pt", wm["img"], wm["adim"]); print(f"[demo] saved BC -> {CKPT}/", flush=True)


def selfcheck(args, device):
    wm = load_wm(f"{CKPT}/wm.pt", device); bc = load_bc(f"{CKPT}/bc.pt", device)
    print(f"[demo] loaded gated WM (arm_px={wm['arm_px']}, OOD={wm['rollout_ood_px']}) + BC", flush=True)
    out = {}
    for method in ("wm", "bc"):
        eng = DemoEngine(wm, bc, device, args.image_size, args.seed, args.action_repeat)
        eng.reset()
        goal = (-0.15, 0.08)                                          # SHIFTED region (x<0) the demos never covered
        eng.set_goal(goal)
        traj, imagined_seen = [], False
        for t in range(args.ep_len):
            o = eng.step(method); traj.append(o["dist_px"]); imagined_seen |= (o["imagined_px"] is not None)
        out[method] = {"final_dist_px": round(traj[-1], 1), "min_dist_px": round(min(traj), 1), "imagined": imagined_seen}
        print(f"  [{method:3}] dragged goal {goal} (SHIFTED) -> final={out[method]['final_dist_px']}px "
              f"min={out[method]['min_dist_px']}px  imagination={'YES' if imagined_seen else 'n/a'}", flush=True)
    money_shot = out["wm"]["min_dist_px"] < 12.0 and out["bc"]["min_dist_px"] > 18.0
    print(f"\n  MONEY SHOT (WM reaches shifted goal, BC fails): {'YES' if money_shot else 'NO'}  "
          f"(wm {out['wm']['min_dist_px']}px vs bc {out['bc']['min_dist_px']}px)", flush=True)
    json.dump(out, open(f"{CKPT}/selfcheck.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/reacher_transitions.npz")
    ap.add_argument("--wm-steps", type=int, default=7000); ap.add_argument("--bc-steps", type=int, default=6000)
    ap.add_argument("--demos", type=int, default=200); ap.add_argument("--ep-len", type=int, default=40)
    ap.add_argument("--action-repeat", type=int, default=2); ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true"); ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.train: train_and_save(args, dev)
    if args.selfcheck: selfcheck(args, dev)
    if not (args.train or args.selfcheck): print("pass --train and/or --selfcheck")


if __name__ == "__main__":
    main()
