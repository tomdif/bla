#!/usr/bin/env python3
"""Probe a custom 3-DOF TORQUE-controlled 3D arm before building the full pipeline: confirm it loads/renders,
the privileged shooting expert actually reaches a 3D target (real torque dynamics, gravity), and measure the
reachable workspace (for normalization + target sampling). Run: MUJOCO_GL=egl python3 -m system1_motion.probe_torque3d
"""
import numpy as np, mujoco

ARM_XML = """
<mujoco model="arm3d">
  <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="256" offheight="256"/></visual>
  <default>
    <joint type="hinge" damping="0.7" armature="0.1" limited="true"/>
    <geom type="capsule" size="0.022" density="600" rgba="0.55 0.6 0.7 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="cam" pos="0.62 -0.62 0.62" xyaxes="0.7 0.7 0 -0.4 0.4 0.82"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" pos="0 0 0" rgba="0.18 0.2 0.24 1"/>
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
    <body name="target" mocap="true" pos="0.3 0.0 0.3">
      <geom name="target" type="sphere" size="0.032" rgba="1 0.32 0.32 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="3.0"/>
    <motor joint="j1" gear="6.0"/>
    <motor joint="j2" gear="4.0"/>
  </actuator>
</mujoco>
"""

m = mujoco.MjModel.from_xml_string(ARM_XML); d = mujoco.MjData(m)
ren = mujoco.Renderer(m, 96, 96)
eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
print("nq", m.nq, "nu", m.nu, "nbody", m.nbody)
print("joint ranges(deg):", np.round(np.degrees(m.jnt_range), 0).tolist())

def ee_pos():   return d.site_xpos[eid].copy()
def tgt_pos():  return d.mocap_pos[0].copy()
def reset(rng):
    mujoco.mj_resetData(m, d)
    d.qpos[:] = rng.uniform(-0.5, 0.5, m.nq); d.qvel[:] = 0; mujoco.mj_forward(m, d)

rng = np.random.RandomState(0); reset(rng)
ren.update_scene(d, camera="cam"); img = ren.render()
print("render shape:", img.shape, "ee:", np.round(ee_pos(), 3))

# reachable workspace over random joint configs -> bounds for normalization + target sampling
P = []
for _ in range(4000):
    d.qpos[:] = rng.uniform(m.jnt_range[:, 0], m.jnt_range[:, 1]); mujoco.mj_forward(m, d); P.append(ee_pos())
P = np.array(P)
print("ee workspace min:", np.round(P.min(0), 3), "max:", np.round(P.max(0), 3))
print("ee workspace (reachable, z>=0.05) min:", np.round(P[P[:,2]>=0.05].min(0),3), "max:", np.round(P[P[:,2]>=0.05].max(0),3))

# privileged shooting expert: try K random torque vectors, sim forward `repeat`, pick min ee->target. Reaches?
def shoot(rng, K=64, repeat=2):
    s = d.qpos.copy(); v = d.qvel.copy(); t = tgt_pos(); best_a, best_dd = None, 1e9
    for _ in range(K):
        a = rng.uniform(-1, 1, m.nu)
        d.qpos[:] = s; d.qvel[:] = v; d.ctrl[:] = a
        for _ in range(repeat): mujoco.mj_step(m, d)
        dd = np.linalg.norm(ee_pos() - t)
        if dd < best_dd: best_dd, best_a = dd, a
    d.qpos[:] = s; d.qvel[:] = v; mujoco.mj_forward(m, d); return best_a

print("\n=== shooting expert reach test (3 random targets) ===")
reach_ok = 0
for trial in range(3):
    reset(rng)
    # sample a target in the reachable shell
    while True:
        c = rng.uniform([0.05,-0.3,0.08],[0.42,0.3,0.45])
        if 0.12 < np.linalg.norm(c) < 0.42: break
    d.mocap_pos[0] = c; mujoco.mj_forward(m, d)
    for step in range(60):
        a = shoot(rng); d.ctrl[:] = a
        for _ in range(2): mujoco.mj_step(m, d)
    fd = np.linalg.norm(ee_pos() - tgt_pos()) * 100
    reach_ok += fd < 5
    print(f"  target={np.round(c,3)} final_ee={np.round(ee_pos(),3)} dist={fd:.1f}cm reached={fd<5}")
print(f"\nEXPERT reaches {reach_ok}/3 within 5cm -> {'CONTROLLABLE' if reach_ok>=2 else 'NEEDS TUNING'}")
print("DONE")
