#!/usr/bin/env python3
"""Probe the HARDER task (clean version): the working reach arm + a FORBIDDEN ZONE (no-go region) between the
ee and the goal. No contact -> dynamics stays the part that already works; the difficulty is purely a NON-CONVEX
planning cost (dist-to-goal + penalty for ee inside the zone). The direct path crosses the zone; escaping the
local minimum needs a DETOUR (temporarily move away from the goal). This is where short-horizon/greedy CEM gets
trapped and a better planner wins -- so it isolates the PLANNER.

Validate with the TRUE simulator as a perfect world model (tests the TASK, not learned dynamics):
  (1) GREEDY short-horizon planning toward (dist+zone) gets TRAPPED at the zone (final far, high zone time),
  (2) LONGER-horizon planning escapes / a DETOUR waypoint reaches,
  (3) with NO zone, greedy reaches (sanity).
Run: MUJOCO_GL=egl python3 -m system1_motion.probe_zone
"""
import numpy as np, mujoco
from system1_motion.r3_torque import Arm, RMIN, RMAX

def zone_pen(ee, zc, zr, W):                                    # smooth penalty bump inside the zone
    return W * np.exp(-(np.linalg.norm(np.asarray(ee) - zc) ** 2) / (2 * zr ** 2))

def cost(ee, goal, zc, zr, W):
    return np.linalg.norm(np.asarray(ee) - goal) + zone_pen(ee, zc, zr, W)

def truesim_mpc(arm, goal, zc, zr, W, horizon, K=200, ep_len=60):
    """perfect-WM CEM-MPC: each step sample K torque seqs, roll in the TRUE sim, score by summed cost over the
    horizon, apply the best first action. Short horizon = greedy. Returns final dist + fraction of time in-zone."""
    in_zone = 0
    for t in range(ep_len):
        s, v = arm.d.qpos.copy(), arm.d.qvel.copy(); best_a, best_c = None, 1e9
        for _ in range(K):
            seq = arm.rng.uniform(-1, 1, (horizon, arm.m.nu)).astype(np.float32)
            arm.d.qpos[:] = s; arm.d.qvel[:] = v; c = 0.0
            for h in range(horizon):
                arm.d.ctrl[:] = seq[h]
                for _ in range(arm.ar): mujoco.mj_step(arm.m, arm.d)
                c += cost(arm.ee(), goal, zc, zr, W)
            if c < best_c: best_c, best_a = c, seq[0]
        arm.d.qpos[:] = s; arm.d.qvel[:] = v; mujoco.mj_forward(arm.m, arm.d)
        arm.step(best_a)
        if np.linalg.norm(arm.ee() - zc) < zr: in_zone += 1
    return float(np.linalg.norm(arm.ee() - goal)) * 100.0, in_zone / ep_len

def detour(arm, goal, zc, zr, ep_len=40):
    """waypoint offset PERPENDICULAR to the zone (go around), then to goal."""
    arm.set_target(goal)
    side = np.cross(goal - zc, np.array([0, 0, 1.0])); side = side / (np.linalg.norm(side) + 1e-9)
    wp = zc + side * (zr + 0.08); wp[2] = max(0.08, zc[2])
    for tgt in (wp, goal):
        arm.set_target(tgt)
        for _ in range(ep_len): arm.step(arm.pd_reach())
    return float(np.linalg.norm(arm.ee() - goal)) * 100.0


def main():
    arm = Arm(0); W, zr = 0.6, 0.10
    # goal that requires moving INWARD from the extended start; zone on the direct path
    goals = [np.array([0.14, 0.06, 0.20]), np.array([0.16, -0.05, 0.24]), np.array([0.13, 0.02, 0.16])]
    print("=== GREEDY (horizon=3) vs LONGER (horizon=12) planning toward dist+zone, + DETOUR, per goal ===")
    for g in goals:
        arm.reset(); start = arm.ee(); zc = 0.5 * (start + g); zc[2] = max(0.10, zc[2])   # zone midway start->goal
        arm.reset(); dg, zg = truesim_mpc(arm.__class__(0), g, zc, zr, W, horizon=3)        # fresh arm each
        a2 = arm.__class__(1); a2.reset(); dl, zl = truesim_mpc(a2, g, zc, zr, W, horizon=12)
        a3 = arm.__class__(2); a3.reset(); dd = detour(a3, g, zc, zr)
        a4 = arm.__class__(3); a4.reset(); dn, zn = truesim_mpc(a4, g, np.array([9, 9, 9.0]), zr, W, horizon=3)  # no zone
        print(f"  goal={np.round(g,3)} zone={np.round(zc,3)}")
        print(f"     greedy(h3):  final={dg:5.1f}cm in_zone={zg:.2f}   longer(h12): final={dl:5.1f}cm in_zone={zl:.2f}")
        print(f"     detour:      final={dd:5.1f}cm                no-zone(h3): final={dn:5.1f}cm (sanity reach)")
    print("\nVERDICT: genuine planner task IF greedy(h3) trapped (far + high in_zone) while longer/detour reach AND no-zone reaches.")
    print("DONE")


if __name__ == "__main__":
    main()
