#!/usr/bin/env python3
"""Probe the HARDER task env: 3-DOF torque arm + a WALL obstacle between the arm and the goal. This is the
structure that exercises the proposal stack (longer-horizon, non-convex: the greedy distance-reducing action
drives the ee into the wall; reaching requires a DETOUR over/around). Before building the full pipeline, validate
the env is a genuine obstacle task:
  (1) a DIRECT reach (straight pd toward the goal) gets BLOCKED by the wall (stays far, hits contacts),
  (2) a DETOUR reach (waypoint over the wall, then down) SUCCEEDS,
  (3) workspace / behind-wall goal region.
A clean gap (direct fails, detour works) = the planner now matters. Run: MUJOCO_GL=egl python3 -m system1_motion.probe_obstacle
"""
import numpy as np, mujoco

OBST_XML = """
<mujoco model="arm3d_wall">
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="256" offheight="256"/></visual>
  <default>
    <joint type="hinge" damping="0.3" armature="0.1" limited="true"/>
    <geom type="capsule" size="0.022" density="600" rgba="0.55 0.6 0.7 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="cam" pos="0.62 -0.62 0.62" xyaxes="0.7 0.7 0 -0.4 0.4 0.82"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" pos="0 0 0" rgba="0.18 0.2 0.24 1"/>
    <!-- WALL obstacle: thin in x, spans y in [-0.10,0.10], z in [0.06,0.34]; arm must clear the top or a side -->
    <geom name="wall" type="box" pos="0.20 0 0.20" size="0.012 0.10 0.14" rgba="0.75 0.45 0.32 1"/>
    <body name="base" pos="0 0 0.06">
      <joint name="j0" axis="0 0 1" range="-180 180"/>
      <geom fromto="0 0 -0.06 0 0 0.06" size="0.03"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="j1" axis="0 1 0" range="-100 100"/>
        <geom fromto="0 0 0 0.2 0 0"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" axis="0 1 0" range="-150 150"/>
          <geom fromto="0 0 0 0.2 0 0"/>
          <body name="ee" pos="0.2 0 0">
            <geom name="fingertip" type="sphere" size="0.028" rgba="0.25 0.95 0.4 1"/>
            <site name="ee" pos="0 0 0" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
    <body name="target" mocap="true" pos="0.3 0.0 0.15">
      <geom name="target" type="sphere" size="0.032" rgba="1 0.32 0.32 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="3.0"/><motor joint="j1" gear="6.0"/><motor joint="j2" gear="4.0"/>
  </actuator>
</mujoco>
"""


class ObstacleArm:
    def __init__(self, seed=0, ar=12):
        self.m = mujoco.MjModel.from_xml_string(OBST_XML); self.d = mujoco.MjData(self.m)
        self.ren = mujoco.Renderer(self.m, 96, 96); self.ar = ar
        self.eid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.wall_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "wall")
        self.rng = np.random.RandomState(seed)
    def ee(self):  return self.d.site_xpos[self.eid].copy()
    def tgt(self): return self.d.mocap_pos[0].copy()
    def reset(self):
        mujoco.mj_resetData(self.m, self.d); self.d.qpos[:] = self.rng.uniform(-0.6, 0.6, self.m.nq); self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
    def set_target(self, xyz): self.d.mocap_pos[0] = xyz; mujoco.mj_forward(self.m, self.d)
    def step(self, ctrl): self.d.ctrl[:] = np.clip(ctrl, -1, 1); [mujoco.mj_step(self.m, self.d) for _ in range(self.ar)]
    def render(self): self.ren.update_scene(self.d, camera="cam"); return self.ren.render()
    def pd_reach(self, tgt=None, Kp=130.0, Kd=16.0):
        t = self.tgt() if tgt is None else np.asarray(tgt); ee = self.ee()
        jacp = np.zeros((3, self.m.nv)); mujoco.mj_jacSite(self.m, self.d, jacp, None, self.eid)
        tau = jacp.T @ (Kp * (t - ee) - Kd * (jacp @ self.d.qvel)) + self.d.qfrc_bias
        return np.clip(tau / self.m.actuator_gear[:, 0], -1, 1).astype(np.float32)
    def wall_contacts(self):                                   # count arm<->wall contacts this step
        return sum(1 for i in range(self.d.ncon) if self.wall_gid in (self.d.contact[i].geom1, self.d.contact[i].geom2))


def run_reach(arm, goal, waypoints, steps_each=22):
    """drive through waypoints (then the goal) with PD; report final dist + whether the wall was hit."""
    arm.reset(); arm.set_target(goal); hits = 0
    for wp in list(waypoints) + [goal]:
        for _ in range(steps_each):
            arm.step(arm.pd_reach(tgt=wp)); hits += arm.wall_contacts()
    return float(np.linalg.norm(arm.ee() - goal)) * 100.0, hits


def main():
    arm = ObstacleArm(0)
    print("nq", arm.m.nq, "ncam", arm.m.ncam, "render", arm.render().shape)
    arm.reset(); print("start ee", np.round(arm.ee(), 3))
    # behind-wall goals: x beyond the wall (>0.24), |y| small (in the wall's shadow) -> need to clear the top
    behind = [np.array([0.30, 0.00, 0.13]), np.array([0.28, 0.05, 0.11]), np.array([0.31, -0.04, 0.16])]
    print("\n=== DIRECT reach (straight pd) -- should be BLOCKED by wall ===")
    for g in behind:
        d, hits = run_reach(arm, g, [])
        print(f"  goal={np.round(g,3)}  final={d:5.1f}cm  wall_hits={hits}  reached={d<5}")
    print("\n=== DETOUR reach (waypoint OVER the wall top, then down) -- should SUCCEED ===")
    for g in behind:
        over = np.array([0.18, g[1], 0.42])                    # lift above the wall (top at z=0.34) at the wall's x
        d, hits = run_reach(arm, g, [over])
        print(f"  goal={np.round(g,3)}  via_over={np.round(over,3)}  final={d:5.1f}cm  wall_hits={hits}  reached={d<5}")
    # workspace sanity: front (non-blocked) goals reachable directly
    print("\n=== control: FRONT goal (no wall in path) direct reach ===")
    for g in [np.array([0.0, 0.28, 0.20]), np.array([0.0, -0.26, 0.18])]:
        d, hits = run_reach(arm, g, [])
        print(f"  goal={np.round(g,3)}  final={d:5.1f}cm  wall_hits={hits}  reached={d<5}")
    print("\nVERDICT: genuine obstacle task IF direct(behind) fails AND detour(behind) succeeds AND front reaches.")
    print("DONE")


if __name__ == "__main__":
    main()
