"""BLA System-1 demo: object-file legibility via full-rollout batched encode.

OF-JEPA v0's identity binding lives WITHIN a single multi-frame
encode_video(T) call (Phase 8C persistent slot_proto + Sinkhorn matching
across the temporal axis). It does NOT survive across independent
single-frame calls. So for legibility demos, we collect the entire env
rollout first, then encode it in ONE call to encode_video, then decode
per-frame from the resulting [T, S, slot_dim] tensor.

See feedback_of_jepa_legibility_requires_temporal_window.md for the
canonical architectural framing.

Three figures, one script:

  A — Object-file trajectory tracks
    Scripted rollout. Encode the whole video in one pass. Plot
    slot_to_pos_aux per-slot decoded trajectories overlaid on ground-
    truth cubeA / cubeB / eef trajectories. Identity-conditioned via
    Hungarian match at t=0.

  B — Per-object surprise (cubeB teleport)
    Same rollout pattern, but teleport cubeB at t=PERTURB_T via mujoco
    set_state. After batched encode, plot the per-slot decoded
    position's step-to-step change. The cubeB-bound slot should spike;
    cubeA and eef slots should stay smooth.

  C — Counterfactual branch from a chosen timestep
    Take slot_states[t_split] from the batched encode. Branch into TWO
    short (H=5) latent rollouts via Phase 17 predictor with different
    action sequences (push +x vs push +y). Render the two predicted
    final cubeA positions to show the predictor IS responsive to
    actions at short horizons (the regime it was trained for).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MUJOCO_GL", "egl")

from system1_jepa.of_jepa import OFJEPAConfig
from system1_jepa.geometry_adapter import ObjectFileGeometryAdapter
from system1_jepa.identity_probe import hungarian_assign
from scripts.slot_jepa_robosuite_train import ActionConditionedOFJEPA
from scripts.phase15_planning import build_env
from scripts.phase16_policy_prior_mpc import rollout_scripted_prior


# ---------- model + adapter ----------
def load_model(args):
    cfg = OFJEPAConfig(
        n_files=args.n_slots, id_dim=args.slot_dim // 2,
        state_dim=args.slot_dim // 2, proposal_dim=args.slot_dim,
    )
    m = ActionConditionedOFJEPA(
        image_size=args.image_size, cfg=cfg,
        action_dim=args.action_dim, use_action=True,
    ).to(args.device)
    m.load_state_dict(torch.load(args.model_action, map_location=args.device))
    m.eval()
    return m


def load_adapter(args, slot_dim_flat):
    ckpt = torch.load(args.adapter_ckpt, map_location=args.device)
    cfg = ckpt.get("config", {})
    adapter = ObjectFileGeometryAdapter(
        slot_dim=int(cfg.get("slot_dim", slot_dim_flat)),
        goal_dim=int(cfg.get("goal_dim", 2)),
        out_dim=int(cfg.get("out_dim", 10)),
        hidden=int(cfg.get("hidden", args.adapter_hidden)),
        n_hidden=int(cfg.get("n_hidden", args.adapter_n_hidden)),
    ).to(args.device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter


# ---------- coords ----------
def norm_xy(p_xy):
    return np.clip((np.asarray(p_xy) + 0.3) / 0.6, 0.0, 1.0).astype(np.float32)


def unnorm_xy(n_xy):
    return (np.asarray(n_xy) * 0.6 - 0.3).astype(np.float32)


def get_cube_xy(obs, name):
    return np.asarray(obs[f"{name}_pos"][:2], dtype=np.float32)


def get_eef_xy(obs):
    return np.asarray(obs["robot0_eef_pos"][:2], dtype=np.float32)


# ---------- env perturbation ----------
def teleport_cube(env, name, new_xy):
    sim = env.sim
    joint_id = None
    for jname in sim.model.joint_names:
        if name.lower() in jname.lower():
            joint_id = sim.model.joint_name2id(jname)
            break
    if joint_id is None:
        raise RuntimeError(f"No mujoco joint found for cube {name}")
    addr = sim.model.jnt_qposadr[joint_id]
    qpos = sim.data.qpos.copy()
    qpos[addr + 0] = float(new_xy[0])
    qpos[addr + 1] = float(new_xy[1])
    sim.data.qpos[:] = qpos
    sim.forward()
    return env._get_observations()


# ---------- collection ----------
def collect_episode(env, plan_actions, jepa_stride, perturb_t=None,
                     perturb_cube="cubeB", perturb_delta=(0.08, 0.08)):
    """Execute plan in env. Record per-step frame + GT positions.

    Returns:
      frames: [T+1, H, W, 3] uint8 (one per env-stride boundary)
      actions: [T, action_dim]
      gt: dict of [T+1, 2] arrays for cubeA, cubeB, eef
    """
    obs = env.reset()
    frames = [obs["agentview_image"].copy()]
    gt = {"cubeA": [get_cube_xy(obs, "cubeA")],
           "cubeB": [get_cube_xy(obs, "cubeB")],
           "eef": [get_eef_xy(obs)]}
    perturb_record = None

    for t, a in enumerate(plan_actions):
        if perturb_t is not None and t == perturb_t:
            before = get_cube_xy(obs, perturb_cube).copy()
            tele_xy = before + np.array(perturb_delta, dtype=np.float32)
            obs = teleport_cube(env, perturb_cube, tele_xy)
            perturb_record = {"t": int(t), "cube": perturb_cube,
                                "before_xy": before.tolist(),
                                "after_xy": get_cube_xy(obs, perturb_cube).tolist()}
        for _ in range(jepa_stride):
            obs, _, _, _ = env.step(a)
        frames.append(obs["agentview_image"].copy())
        gt["cubeA"].append(get_cube_xy(obs, "cubeA"))
        gt["cubeB"].append(get_cube_xy(obs, "cubeB"))
        gt["eef"].append(get_eef_xy(obs))

    return {
        "frames": np.stack(frames, axis=0),
        "actions": np.asarray(plan_actions, dtype=np.float32),
        "gt": {k: np.stack(v, axis=0) for k, v in gt.items()},
        "perturb": perturb_record,
    }


# ---------- batched encode ----------
@torch.no_grad()
def encode_full_rollout(model, frames_TWHC):
    """Encode the full T-frame sequence in ONE encode_video call.

    Returns slot_states [T, S, slot_dim] (torch on model device).
    """
    device = next(model.parameters()).device
    video = (torch.from_numpy(frames_TWHC).permute(0, 3, 1, 2).float() / 255.0
              ).to(device)  # [T, 3, H, W]
    slot_states, _ = model.encode_video(video)
    return slot_states


@torch.no_grad()
def decode_positions_per_frame(model, slot_states):
    """slot_states [T, S, slot_dim] → [T, S, 2] normalized positions."""
    return model.slot_to_pos_aux(slot_states).detach().cpu().numpy()


def assign_slot_to_entity(decoded_pos_t0_S2, entity_pos_norm_N2):
    """Hungarian-match at t=0. Returns slot index for each entity in order."""
    rows, cols, _ = hungarian_assign(decoded_pos_t0_S2, entity_pos_norm_N2)
    out = [None] * len(entity_pos_norm_N2)
    for r, c in zip(rows.tolist(), cols.tolist()):
        if 0 <= c < len(entity_pos_norm_N2):
            out[c] = int(r)
    return out


# ---------- latent rollout (predictor) ----------
@torch.no_grad()
def latent_rollout(model, init_slot, action_seq_np):
    """init_slot [S, slot_dim]. Returns slot trajectory [H+1, S, slot_dim]."""
    device = next(model.parameters()).device
    id_dim = model.cfg.id_dim
    traj = [init_slot.clone()]
    slot = init_slot.clone()
    for a in action_seq_np:
        a_t = torch.from_numpy(a).float().to(device).unsqueeze(0)
        state_pred = model.predict_state_delta(slot.unsqueeze(0), a_t)[0]
        slot = torch.cat([slot[:, :id_dim], state_pred], dim=-1)
        traj.append(slot.clone())
    return torch.stack(traj, dim=0)


# ---------- demo runners ----------
def run_demo_a(env, model, args, out_dir):
    """Object-file trajectory tracks via batched encode."""
    obs = env.reset()
    cubeA0 = get_cube_xy(obs, "cubeA")
    push_dist = 0.12
    goal_world = cubeA0 + np.array([0.0, +push_dist], dtype=np.float32)
    plan_actions = rollout_scripted_prior(
        env, obs, goal_world, args.plan_horizon, args.jepa_stride,
    )
    episode = collect_episode(env, plan_actions, args.jepa_stride)
    slot_states = encode_full_rollout(model, episode["frames"])
    decoded = decode_positions_per_frame(model, slot_states)  # [T+1, S, 2] norm

    # Identity-bind at t=0
    init_decoded = decoded[0]   # [S, 2]
    init_gt_norm = np.stack([
        norm_xy(episode["gt"]["cubeA"][0]),
        norm_xy(episode["gt"]["cubeB"][0]),
        norm_xy(episode["gt"]["eef"][0]),
    ])
    slot_for = assign_slot_to_entity(init_decoded, init_gt_norm)
    idx_A, idx_B, idx_eef = slot_for

    # Decoded trajectories per identity, in world coords
    traj_A_dec = unnorm_xy(decoded[:, idx_A])
    traj_B_dec = unnorm_xy(decoded[:, idx_B])
    traj_eef_dec = unnorm_xy(decoded[:, idx_eef])
    traj_A_gt = episode["gt"]["cubeA"]
    traj_B_gt = episode["gt"]["cubeB"]
    traj_eef_gt = episode["gt"]["eef"]

    # Decode errors
    err_A = float(np.linalg.norm(traj_A_dec - traj_A_gt, axis=-1).mean())
    err_B = float(np.linalg.norm(traj_B_dec - traj_B_gt, axis=-1).mean())
    err_eef = float(np.linalg.norm(traj_eef_dec - traj_eef_gt, axis=-1).mean())
    actual_drift_A = float(np.linalg.norm(traj_A_gt[-1] - traj_A_gt[0]))
    decoded_drift_A = float(np.linalg.norm(traj_A_dec[-1] - traj_A_dec[0]))

    print(json.dumps({"event": "demo_a_diag",
                       "slot_indices": {"cubeA": idx_A, "cubeB": idx_B, "eef": idx_eef},
                       "mean_decode_err_cubeA_m": err_A,
                       "mean_decode_err_cubeB_m": err_B,
                       "mean_decode_err_eef_m": err_eef,
                       "actual_cubeA_drift_m": actual_drift_A,
                       "decoded_cubeA_drift_m": decoded_drift_A,
                       "n_frames": int(decoded.shape[0])}),
           flush=True)

    # Render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(episode["frames"][0])
    axes[0].set_title("Initial scene (t=0)")
    axes[0].axis("off")

    ax = axes[1]
    ax.set_xlim(-0.25, 0.25); ax.set_ylim(-0.25, 0.25)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    # GT solid, decoded dashed
    for (gt_traj, dec_traj, color, label) in [
        (traj_A_gt, traj_A_dec, "red",   "cubeA"),
        (traj_B_gt, traj_B_dec, "blue",  "cubeB"),
        (traj_eef_gt, traj_eef_dec, "green", "eef"),
    ]:
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], color=color, lw=2.0, alpha=0.85,
                  label=f"{label} actual")
        ax.plot(dec_traj[:, 0], dec_traj[:, 1], color=color, lw=1.5, ls="--",
                  alpha=0.6, label=f"{label} object-file decoded")
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c=color, marker="o", s=70,
                    edgecolors="black", linewidths=0.5)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c=color, marker="X", s=110,
                    edgecolors="black", linewidths=0.5)
    ax.scatter(goal_world[0], goal_world[1],
                c="orange", marker="*", s=220, edgecolors="black", linewidths=0.7,
                label="goal")
    ax.set_title(f"Object-file trajectory tracks (batched encode)\n"
                  f"mean decode err: cubeA {err_A*100:.1f}cm, "
                  f"cubeB {err_B*100:.1f}cm, eef {err_eef*100:.1f}cm")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", fontsize=7)

    out_png = Path(out_dir) / "demo_A_trajectory_tracks.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "demo": "A_trajectory_tracks",
        "slot_indices": {"cubeA": idx_A, "cubeB": idx_B, "eef": idx_eef},
        "mean_decode_err_m": {"cubeA": err_A, "cubeB": err_B, "eef": err_eef},
        "actual_cubeA_drift_m": actual_drift_A,
        "decoded_cubeA_drift_m": decoded_drift_A,
        "png": str(out_png),
    }


def run_demo_b(env, model, args, out_dir):
    """Per-object surprise: teleport cubeB mid-rollout; cubeB slot decoded
    position should jump at the perturbation."""
    obs = env.reset()
    cubeA0 = get_cube_xy(obs, "cubeA")
    push_dist = 0.12
    goal_world = cubeA0 + np.array([+push_dist, 0.0], dtype=np.float32)
    plan_actions = rollout_scripted_prior(
        env, obs, goal_world, args.demo_b_horizon, args.jepa_stride,
    )
    episode = collect_episode(env, plan_actions, args.jepa_stride,
                                perturb_t=args.demo_b_perturb_t,
                                perturb_cube="cubeB",
                                perturb_delta=(0.08, 0.08))
    slot_states = encode_full_rollout(model, episode["frames"])
    decoded = decode_positions_per_frame(model, slot_states)  # [T+1, S, 2] norm

    # Identity-bind at t=0
    init_gt_norm = np.stack([
        norm_xy(episode["gt"]["cubeA"][0]),
        norm_xy(episode["gt"]["cubeB"][0]),
        norm_xy(episode["gt"]["eef"][0]),
    ])
    slot_for = assign_slot_to_entity(decoded[0], init_gt_norm)
    idx_A, idx_B, idx_eef = slot_for

    # Per-slot decoded position step-to-step change in cm
    step_change_norm = np.linalg.norm(np.diff(decoded, axis=0), axis=-1)  # [T, S]
    step_change_cm = step_change_norm * 0.6 * 100
    # And per-slot displacement from t=0 in cm
    delta_from_0_cm = np.linalg.norm(decoded - decoded[0:1], axis=-1) * 0.6 * 100

    perturb_t = args.demo_b_perturb_t
    cubeB_slot_step_at_perturb = float(step_change_cm[perturb_t, idx_B])
    cubeA_slot_step_at_perturb = float(step_change_cm[perturb_t, idx_A])
    actual_cubeB_step = float(np.linalg.norm(
        episode["gt"]["cubeB"][perturb_t + 1] - episode["gt"]["cubeB"][perturb_t]
    ) * 100)

    print(json.dumps({"event": "demo_b_diag",
                       "slot_indices": {"cubeA": idx_A, "cubeB": idx_B, "eef": idx_eef},
                       "perturb_t": perturb_t,
                       "actual_cubeB_step_at_perturb_cm": actual_cubeB_step,
                       "cubeB_slot_decoded_step_at_perturb_cm": cubeB_slot_step_at_perturb,
                       "cubeA_slot_decoded_step_at_perturb_cm": cubeA_slot_step_at_perturb,
                       "step_change_cm_at_perturb_per_slot": step_change_cm[perturb_t].tolist()}),
           flush=True)

    # Render: per-slot step-to-step decoded change, with perturb marker
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel L: step-to-step change per slot (the "surprise" line)
    ax = axes[0]
    timesteps = np.arange(step_change_cm.shape[0])
    cmap = plt.get_cmap("tab10")
    labels = {idx_A: "cubeA", idx_B: "cubeB", idx_eef: "eef"}
    for s in range(args.n_slots):
        if s in labels:
            lab = f"slot {s} ({labels[s]})"; lw = 2.5
        else:
            lab = f"slot {s} (unbound)"; lw = 1.0
        ax.plot(timesteps, step_change_cm[:, s], color=cmap(s % 10),
                  lw=lw, label=lab)
    mean_line = step_change_cm.mean(axis=1)
    ax.plot(timesteps, mean_line, color="black", lw=2.5, ls="--",
              label="mean across slots (frame-level baseline)")
    ax.axvline(perturb_t + 0.5, color="red", ls=":", lw=2,
                label=f"cubeB teleport (+8,+8cm) @ t={perturb_t}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("step-to-step decoded slot Δxy (cm)")
    ax.set_title("Per-object surprise: which slot spikes?\n"
                  f"cubeB-slot Δ at perturb: {cubeB_slot_step_at_perturb:.1f}cm, "
                  f"cubeA-slot: {cubeA_slot_step_at_perturb:.1f}cm")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel R: cumulative displacement from t=0
    ax = axes[1]
    for s in range(args.n_slots):
        if s in labels:
            lab = f"slot {s} ({labels[s]})"; lw = 2.5
        else:
            lab = f"slot {s} (unbound)"; lw = 1.0
        ax.plot(np.arange(delta_from_0_cm.shape[0]),
                  delta_from_0_cm[:, s], color=cmap(s % 10),
                  lw=lw, label=lab)
    ax.axvline(perturb_t + 0.5, color="red", ls=":", lw=2,
                label=f"cubeB teleport @ t={perturb_t}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("cumulative decoded |Δxy| from t=0 (cm)")
    ax.set_title("Cumulative slot displacement")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    out_png = Path(out_dir) / "demo_B_surprise.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "demo": "B_per_object_surprise",
        "slot_indices": {"cubeA": idx_A, "cubeB": idx_B, "eef": idx_eef},
        "perturb_t": perturb_t,
        "actual_cubeB_step_at_perturb_cm": actual_cubeB_step,
        "cubeB_slot_decoded_step_at_perturb_cm": cubeB_slot_step_at_perturb,
        "cubeA_slot_decoded_step_at_perturb_cm": cubeA_slot_step_at_perturb,
        "png": str(out_png),
    }


def run_demo_c(env, model, args, out_dir):
    """Counterfactual: from t_split mid-episode, two short latent branches."""
    obs = env.reset()
    cubeA0 = get_cube_xy(obs, "cubeA")
    push_dist = 0.10
    goal_world = cubeA0 + np.array([0.0, +push_dist], dtype=np.float32)
    plan_actions = rollout_scripted_prior(
        env, obs, goal_world, args.plan_horizon, args.jepa_stride,
    )
    episode = collect_episode(env, plan_actions, args.jepa_stride)
    slot_states = encode_full_rollout(model, episode["frames"])
    decoded = decode_positions_per_frame(model, slot_states)

    init_gt_norm = np.stack([
        norm_xy(episode["gt"]["cubeA"][0]),
        norm_xy(episode["gt"]["cubeB"][0]),
        norm_xy(episode["gt"]["eef"][0]),
    ])
    slot_for = assign_slot_to_entity(decoded[0], init_gt_norm)
    idx_A, idx_B, idx_eef = slot_for

    # Pick split point: where cubeA is being approached (eef near cube)
    t_split = args.counterfactual_t_split
    if t_split >= slot_states.shape[0]:
        t_split = slot_states.shape[0] - 1
    init_slot = slot_states[t_split]   # [S, slot_dim]

    # Two counterfactual short action sequences (H=5 — predictor's training regime)
    H = args.counterfactual_horizon
    action_dim = args.action_dim
    actions_pushX = np.zeros((H, action_dim), dtype=np.float32)
    actions_pushX[:, 0] = +1.0;  actions_pushX[:, 2] = -0.2;  actions_pushX[:, 6] = +1.0
    actions_pushY = np.zeros((H, action_dim), dtype=np.float32)
    actions_pushY[:, 1] = +1.0;  actions_pushY[:, 2] = -0.2;  actions_pushY[:, 6] = +1.0

    traj_X = latent_rollout(model, init_slot, actions_pushX)
    traj_Y = latent_rollout(model, init_slot, actions_pushY)
    pos_X = decode_positions_per_frame(model, traj_X)   # [H+1, S, 2]
    pos_Y = decode_positions_per_frame(model, traj_Y)
    cubeA_traj_X = unnorm_xy(pos_X[:, idx_A])
    cubeA_traj_Y = unnorm_xy(pos_Y[:, idx_A])
    div_X = float(np.linalg.norm(cubeA_traj_X[-1] - cubeA_traj_X[0]))
    div_Y = float(np.linalg.norm(cubeA_traj_Y[-1] - cubeA_traj_Y[0]))
    branch_separation = float(np.linalg.norm(cubeA_traj_X[-1] - cubeA_traj_Y[-1]))
    print(json.dumps({"event": "demo_c_diag",
                       "t_split": t_split,
                       "counterfactual_horizon": H,
                       "pushX_cubeA_drift_m": div_X,
                       "pushY_cubeA_drift_m": div_Y,
                       "branch_separation_m": branch_separation}),
           flush=True)

    # Render: scene at t_split + counterfactual branches
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(episode["frames"][t_split])
    axes[0].set_title(f"Scene at t_split = {t_split}\n(branch from here)")
    axes[0].axis("off")

    ax = axes[1]
    cubeA_split_world = unnorm_xy(decoded[t_split, idx_A])
    ax.set_xlim(cubeA_split_world[0] - 0.10, cubeA_split_world[0] + 0.10)
    ax.set_ylim(cubeA_split_world[1] - 0.10, cubeA_split_world[1] + 0.10)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.plot(cubeA_traj_X[:, 0], cubeA_traj_X[:, 1],
              "b-", lw=2.5, alpha=0.85, label="imagine push +x")
    ax.plot(cubeA_traj_Y[:, 0], cubeA_traj_Y[:, 1],
              "g-", lw=2.5, alpha=0.85, label="imagine push +y")
    ax.scatter(cubeA_split_world[0], cubeA_split_world[1],
                c="black", marker="o", s=80, edgecolors="black",
                linewidths=0.7, label="cubeA at t_split")
    ax.scatter(cubeA_traj_X[-1, 0], cubeA_traj_X[-1, 1],
                c="blue", marker="X", s=140, edgecolors="black", linewidths=0.7)
    ax.scatter(cubeA_traj_Y[-1, 0], cubeA_traj_Y[-1, 1],
                c="green", marker="X", s=140, edgecolors="black", linewidths=0.7)
    ax.set_title(f"Counterfactual cubeA branches (H={H} latent steps)\n"
                  f"branch separation: {branch_separation*100:.1f}cm")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="lower left", fontsize=8)

    out_png = Path(out_dir) / "demo_C_counterfactual.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "demo": "C_counterfactual_branch",
        "t_split": t_split,
        "counterfactual_horizon": H,
        "pushX_cubeA_drift_m": div_X,
        "pushY_cubeA_drift_m": div_Y,
        "branch_separation_m": branch_separation,
        "png": str(out_png),
    }


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-action", required=True)
    p.add_argument("--adapter-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--slot-dim", type=int, default=128)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--jepa-stride", type=int, default=4)
    p.add_argument("--adapter-hidden", type=int, default=256)
    p.add_argument("--adapter-n-hidden", type=int, default=3)
    # Demo A
    p.add_argument("--plan-horizon", type=int, default=25)
    # Demo B
    p.add_argument("--demo-b-horizon", type=int, default=30)
    p.add_argument("--demo-b-perturb-t", type=int, default=15)
    # Demo C
    p.add_argument("--counterfactual-t-split", type=int, default=12)
    p.add_argument("--counterfactual-horizon", type=int, default=5)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model = load_model(args)
    _ = load_adapter(args, slot_dim_flat=args.n_slots * args.slot_dim)  # validation only

    big_horizon = max(args.plan_horizon, args.demo_b_horizon) * args.jepa_stride * 4 + 200
    env_a = build_env(args.image_size, horizon=big_horizon)
    sum_a = run_demo_a(env_a, model, args, out)
    env_b = build_env(args.image_size, horizon=big_horizon)
    sum_b = run_demo_b(env_b, model, args, out)
    env_c = build_env(args.image_size, horizon=big_horizon)
    sum_c = run_demo_c(env_c, model, args, out)

    summary = {"demo_a": sum_a, "demo_b": sum_b, "demo_c": sum_c, "seed": args.seed}
    with open(Path(out) / "demo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"event": "all_done", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
