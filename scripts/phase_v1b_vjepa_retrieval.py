"""Phase V1b — V-JEPA 2 clip-summary as DemoRetriever key on PickPlaceCan.

Tests whether V-JEPA 2's clip-summary feature (1024-D mean-pooled over
patch tokens) is a better retrieval key than DR1's engineered 6-D
geometry key (can_xy, eef_xy, can_z, eef_z).

Protocol (mirrors DR1; differs in render resolution and reset pool):
  - robosuite PickPlaceCan rendered at 256×256 (V-JEPA 2 native)
  - Demo bank = 24 working demos screened from 100 (state-matched
    reset → lift z_gain ≥ 0.05m)
  - Reset target: RANDOM from full 100-demo pool (matches DR3 protocol
    — creates NN distance > 0 for ~76% of episodes where geometry
    retrieval will actually have to choose)
  - Modes:
      demo_no_cem_oracle           ceiling (uses reset target's own actions)
      demo_no_cem_cycle            broken baseline (first 5 working demos)
      geometry_top1                DR1 baseline (6-D mujoco pose key)
      vjepa_top1                   NEW: V-JEPA 2 clip-summary key
      vjepa_top3_avg               NEW: top-3 averaged

Retrieval key construction (both bank and query):
  - Reset env to a target init state
  - Apply 4 frames of no-op action (gripper open, zero motion)
  - Capture the 4 rendered frames at 256×256
  - Encode with V-JEPA 2 ViT-L; mean-pool ALL patch tokens → 1024-D

Pre-committed gate: vjepa_top1 mean ≥ geometry_top1 mean − 0.02
                   (V-JEPA must match or beat geometry within tolerance)
Strong-pass:        vjepa_top1 mean ≥ geometry_top1 mean + 0.02
                   AND vjepa_top1 std < geometry_top1 std
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from bla.recipes import DemoState, DemoRetriever


# ---------- robosuite + demos ----------
IMAGE_SIZE = 256       # V-JEPA 2 native
LIFT_TARGET_Z_GAIN = 0.10   # cm above table; same as DR1 PickPlaceCan
N_VJEPA_FRAMES = 4     # short clip for V-JEPA 2 encoding


def build_env(image_size: int = IMAGE_SIZE, horizon: int = 400):
    import robosuite as rs
    return rs.make("PickPlaceCan", robots="Panda",
                    has_renderer=False, has_offscreen_renderer=True,
                    use_camera_obs=True, use_object_obs=True,
                    camera_names="agentview",
                    camera_heights=image_size, camera_widths=image_size,
                    horizon=horizon)


def reset_env_to_demo_init(env, init_state: np.ndarray):
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    return env._get_observations()


def read_mujoco_pose(env) -> np.ndarray:
    """6-D: [can_x, can_y, eef_x, eef_y, can_z, eef_z] from sim.data."""
    sim = env.sim
    can = sim.data.get_body_xpos("Can_main").copy()
    eef = sim.data.get_body_xpos("gripper0_right_eef").copy()
    return np.concatenate([can[:2], eef[:2], [can[2], eef[2]]]).astype(np.float32)


def get_can_z(env) -> float:
    return float(env.sim.data.get_body_xpos("Can_main")[2])


def extract_demos_to_cache(source: str, out_dir: str, n_demos: int = 100):
    """Extract actions + init_state per demo from robomimic hdf5."""
    import h5py
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    needs_rebuild = any(not (out / f"ep_{i:05d}.npz").exists() for i in range(n_demos))
    if not needs_rebuild:
        return
    f = h5py.File(source, "r")
    demos = sorted(f["data"].keys(),
                     key=lambda k: int(k.split("_")[1]))[:n_demos]
    for i, dname in enumerate(demos):
        actions = f[f"data/{dname}/actions"][:].astype(np.float32)
        init_state = f[f"data/{dname}/states"][0].astype(np.float64)
        np.savez(out / f"ep_{i:05d}.npz", actions=actions, init_state=init_state)
    f.close()


def load_demo_data(demo_dir: str, n_demos: int = 100) -> list[dict]:
    """Load all demo records: actions + init_state."""
    records = []
    for i in range(n_demos):
        d = np.load(Path(demo_dir) / f"ep_{i:05d}.npz")
        records.append({"demo_id": i,
                          "actions": d["actions"].astype(np.float32),
                          "init_state": d["init_state"]})
    return records


def screen_working_demos(env, demos: list[dict], n_eval_steps: int = 120,
                            z_gain_threshold: float = 0.05) -> list[int]:
    """Find demos that lift the can ≥ z_gain_threshold on their own init."""
    working = []
    for d in demos:
        obs = reset_env_to_demo_init(env, d["init_state"])
        can_z0 = get_can_z(env)
        n = min(len(d["actions"]), n_eval_steps)
        for a in d["actions"][:n]:
            env.step(a)
        z_gain = get_can_z(env) - can_z0
        if z_gain >= z_gain_threshold:
            working.append(d["demo_id"])
    return working


# ---------- V-JEPA 2 ----------
class VJEPA2Encoder:
    def __init__(self, device: str = "cuda"):
        from transformers import VJEPA2Model, AutoVideoProcessor
        m_id = "facebook/vjepa2-vitl-fpc64-256"
        print(json.dumps({"event": "loading_vjepa2", "model_id": m_id}),
              flush=True)
        self.processor = AutoVideoProcessor.from_pretrained(m_id)
        self.model = VJEPA2Model.from_pretrained(
            m_id, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode_clip(self, frames_THWC: np.ndarray) -> np.ndarray:
        """frames: [T, H, W, 3] uint8. Returns 1024-D mean-pooled key."""
        inputs = self.processor(videos=list(frames_THWC),
                                  return_tensors="pt")
        pv = inputs["pixel_values_videos"].to(self.device).to(torch.bfloat16)
        out = self.model(pv, skip_predictor=True)
        # last_hidden_state: [B, N_tokens, D]
        clip = out.last_hidden_state[0].mean(dim=0).float().cpu().numpy()
        return clip.astype(np.float32)


# ---------- bank build ----------
def collect_init_clip(env, init_state, n_frames: int = N_VJEPA_FRAMES,
                        action_dim: int = 7) -> np.ndarray:
    """Reset env to init_state; apply n_frames no-op actions; capture
    rendered frames at IMAGE_SIZE. Returns [n_frames, H, W, 3] uint8."""
    obs = reset_env_to_demo_init(env, init_state)
    a_noop = np.zeros(action_dim, dtype=np.float32)
    a_noop[6] = 1.0  # gripper open
    frames = [obs["agentview_image"]]
    for _ in range(n_frames - 1):
        obs, _, _, _ = env.step(a_noop)
        frames.append(obs["agentview_image"])
    return np.stack(frames, axis=0)


def build_retrievers(env, demos: list[dict], working_ids: list[int],
                        vjepa: VJEPA2Encoder) -> tuple[DemoRetriever, DemoRetriever]:
    """Build geometry + V-JEPA retrievers from the same demo bank.

    Geometry key: 6-D mujoco pose at init state.
    V-JEPA key:   1024-D clip-summary from 4-frame init rollout.
    """
    geom_recs, vjepa_recs = [], []
    for demo_id in working_ids:
        d = [r for r in demos if r["demo_id"] == demo_id][0]
        # Geometry key: just reset and read pose
        reset_env_to_demo_init(env, d["init_state"])
        geom_key = read_mujoco_pose(env)
        # V-JEPA key: collect 4-frame clip and encode
        clip_frames = collect_init_clip(env, d["init_state"])
        vjepa_key = vjepa.encode_clip(clip_frames)
        # Outcome score: actual z-gain on its own init (DR3 protocol)
        reset_env_to_demo_init(env, d["init_state"])
        can_z0 = get_can_z(env)
        n = min(len(d["actions"]), 120)
        for a in d["actions"][:n]:
            env.step(a)
        outcome = get_can_z(env) - can_z0
        # Build records
        common = dict(action_seq=d["actions"], init_state=d["init_state"],
                         demo_id=int(demo_id), outcome_score=float(outcome))
        geom_recs.append(DemoState(key=geom_key, **common))
        vjepa_recs.append(DemoState(key=vjepa_key, **common))
        print(json.dumps({"event": "bank_demo", "demo_id": int(demo_id),
                           "outcome_z_gain": float(outcome),
                           "geom_key_dim": geom_key.shape[0],
                           "vjepa_key_dim": vjepa_key.shape[0]}), flush=True)
    rg = DemoRetriever(); rg.build_index(geom_recs)
    rv = DemoRetriever(); rv.build_index(vjepa_recs)
    return rg, rv


# ---------- eval ----------
def get_demo_prior_actions(actions: np.ndarray, H: int, stride: int) -> np.ndarray:
    out = []
    for t in range(H):
        out.append(actions[min(t * stride, len(actions) - 1)])
    return np.stack(out, axis=0)


def run_episode(env, mode: str, reset_demo: dict, demos: list[dict],
                  cycle_ids: tuple, retriever_geom: DemoRetriever,
                  retriever_vjepa: DemoRetriever, vjepa: VJEPA2Encoder,
                  ep_id: int, plan_horizon: int, jepa_stride: int) -> dict:
    obs = reset_env_to_demo_init(env, reset_demo["init_state"])
    can_z0 = get_can_z(env)

    # Build query keys before executing
    geom_q = read_mujoco_pose(env)
    # vjepa_q: re-collect 4-frame init clip; need to reset again because
    # collect_init_clip steps the env. Capture, then reset.
    clip_frames = collect_init_clip(env, reset_demo["init_state"])
    vjepa_q = vjepa.encode_clip(clip_frames)
    obs = reset_env_to_demo_init(env, reset_demo["init_state"])

    # Pick action sequence per mode
    if mode == "demo_no_cem_oracle":
        chosen_id = reset_demo["demo_id"]
        actions_executed = get_demo_prior_actions(
            reset_demo["actions"], plan_horizon, jepa_stride)
    elif mode == "demo_no_cem_cycle":
        chosen_id = cycle_ids[ep_id % len(cycle_ids)]
        d = [r for r in demos if r["demo_id"] == chosen_id][0]
        actions_executed = get_demo_prior_actions(
            d["actions"], plan_horizon, jepa_stride)
    elif mode == "geometry_top1":
        top = retriever_geom.retrieve(geom_q, k=1)[0]
        chosen_id = top.demo_id
        actions_executed = get_demo_prior_actions(
            top.action_seq, plan_horizon, jepa_stride)
    elif mode == "vjepa_top1":
        top = retriever_vjepa.retrieve(vjepa_q, k=1)[0]
        chosen_id = top.demo_id
        actions_executed = get_demo_prior_actions(
            top.action_seq, plan_horizon, jepa_stride)
    elif mode == "vjepa_top3_avg":
        top3 = retriever_vjepa.retrieve(vjepa_q, k=3)
        chosen_id = -1
        seqs = [get_demo_prior_actions(d.action_seq, plan_horizon, jepa_stride)
                  for d in top3]
        actions_executed = np.stack(seqs, 0).mean(0).astype(np.float32)
    elif mode == "geometry_top3_avg":
        top3 = retriever_geom.retrieve(geom_q, k=3)
        chosen_id = -1
        seqs = [get_demo_prior_actions(d.action_seq, plan_horizon, jepa_stride)
                  for d in top3]
        actions_executed = np.stack(seqs, 0).mean(0).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Execute
    for a in actions_executed:
        for _ in range(jepa_stride):
            obs, _, _, _ = env.step(a)
    can_z_end = get_can_z(env)
    z_gain = can_z_end - can_z0
    gain_clipped = max(0.0, z_gain)
    imp = float(min(1.0, gain_clipped / LIFT_TARGET_Z_GAIN))
    success = z_gain >= 0.10

    return {"ep_id": ep_id, "mode": mode,
              "reset_demo_id": int(reset_demo["demo_id"]),
              "chose_demo_id": int(chosen_id),
              "matches_reset_target": bool(chosen_id == int(reset_demo["demo_id"])),
              "can_z0": can_z0, "can_z_end": can_z_end,
              "z_gain_m": z_gain, "improvement": imp,
              "success": int(success)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    p.add_argument("--n-eval-episodes", type=int, default=30)
    p.add_argument("--plan-horizon", type=int, default=30)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--cache-bank", type=str,
                    default="/root/v1b_bank_cache.npz",
                    help="Cache file for the V-JEPA bank (reuse across seeds)")
    p.add_argument("--demo-src",
                    default="/workspace/robomimic_data/can_demo_v141.hdf5")
    p.add_argument("--demo-cache-dir",
                    default="/workspace/robomimic_can_replay")
    p.add_argument("--working-ids",
                    default="5,8,10,13,16,23,25,28,30,41,45,46,47,58,63,66,67,69,81,82,86,90,94,96",
                    help="Known working demo IDs from DR1 (24 demos)")
    p.add_argument("--reset-pool-size", type=int, default=100,
                    help="Reset target sampled from this many demos (DR3 protocol = 100)")
    p.add_argument("--modes", type=str,
                    default="demo_no_cem_oracle,demo_no_cem_cycle,"
                            "geometry_top1,vjepa_top1,vjepa_top3_avg,geometry_top3_avg")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "args", "args": vars(args)}), flush=True)

    # 1. Demo extraction
    extract_demos_to_cache(args.demo_src, args.demo_cache_dir,
                              n_demos=args.reset_pool_size)
    demos = load_demo_data(args.demo_cache_dir, n_demos=args.reset_pool_size)
    working_ids = [int(x) for x in args.working_ids.split(",")]
    cycle_ids = tuple(working_ids[:5])
    print(json.dumps({"event": "demos_loaded",
                       "n_demos": len(demos),
                       "n_working": len(working_ids),
                       "cycle_ids": list(cycle_ids)}), flush=True)

    big_horizon = args.plan_horizon * args.jepa_stride * 4 + 400
    env = build_env(IMAGE_SIZE, big_horizon)

    # 2. Build retrievers (with V-JEPA 2 cache)
    cache_path = Path(args.cache_bank)
    use_cache = cache_path.exists()
    if use_cache:
        print(json.dumps({"event": "loading_bank_cache",
                           "path": str(cache_path)}), flush=True)
        cache = np.load(cache_path, allow_pickle=True)
        geom_keys = cache["geom_keys"]
        vjepa_keys = cache["vjepa_keys"]
        outcomes = cache["outcomes"]
        bank_demo_ids = cache["demo_ids"].tolist()
        vjepa = VJEPA2Encoder(args.device)
        # Reconstruct retrievers
        geom_recs, vjepa_recs = [], []
        for i, did in enumerate(bank_demo_ids):
            d = [r for r in demos if r["demo_id"] == did][0]
            common = dict(action_seq=d["actions"], init_state=d["init_state"],
                            demo_id=int(did), outcome_score=float(outcomes[i]))
            geom_recs.append(DemoState(key=geom_keys[i], **common))
            vjepa_recs.append(DemoState(key=vjepa_keys[i], **common))
        rg = DemoRetriever(); rg.build_index(geom_recs)
        rv = DemoRetriever(); rv.build_index(vjepa_recs)
    else:
        vjepa = VJEPA2Encoder(args.device)
        t0 = time.time()
        rg, rv = build_retrievers(env, demos, working_ids, vjepa)
        # Cache for future seeds
        gk = rg.index_keys(); vk = rv.index_keys()
        outs = np.array([d.outcome_score for d in rg._bank], dtype=np.float32)
        np.savez(cache_path, geom_keys=gk, vjepa_keys=vk, outcomes=outs,
                    demo_ids=np.array(working_ids, dtype=int))
        print(json.dumps({"event": "bank_built",
                           "n_demos": len(rg),
                           "elapsed_s": round(time.time() - t0, 1),
                           "vjepa_key_dim": int(vk.shape[1]),
                           "geom_key_dim": int(gk.shape[1])}), flush=True)

    # 3. Sample reset targets (seed-controlled; DR3 protocol — full pool)
    rng = np.random.RandomState(args.seed)
    reset_ids = list(rng.choice(args.reset_pool_size,
                                   size=args.n_eval_episodes, replace=True))

    # 4. Run modes
    modes = args.modes.split(",")
    all_results = {}
    for mode in modes:
        per_ep = []
        for i, rid in enumerate(reset_ids):
            reset_demo = [d for d in demos if d["demo_id"] == int(rid)][0]
            t0 = time.time()
            r = run_episode(env, mode, reset_demo, demos, cycle_ids,
                              rg, rv, vjepa, i,
                              args.plan_horizon, args.jepa_stride)
            r["elapsed_s"] = time.time() - t0
            per_ep.append(r)
            print(json.dumps({"event": "ep", **r}), flush=True)
        imp_mean = float(np.mean([e["improvement"] for e in per_ep]))
        succ_mean = float(np.mean([e["success"] for e in per_ep]))
        match_mean = float(np.mean([e["matches_reset_target"] for e in per_ep]))
        all_results[mode] = {"per_ep": per_ep,
                                "improvement_mean": imp_mean,
                                "success_rate": succ_mean,
                                "retrieval_match_rate": match_mean}
        print(json.dumps({"event": "mode_done", "mode": mode,
                           "improvement_mean": imp_mean,
                           "success_rate": succ_mean,
                           "retrieval_match_rate": match_mean}), flush=True)

    # 5. Save
    summary = {"args": vars(args),
                  "reset_ids": [int(x) for x in reset_ids],
                  "working_ids": working_ids,
                  "cycle_ids": list(cycle_ids),
                  "results": all_results}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"event": "done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
