#!/usr/bin/env python3
"""M0 -- multi-object push env + probe. Top-down tabletop: a velocity-controlled PUSHER (trivial control) + 3
free-sliding PUCKS (target=green + 2 decoys) + a goal zone. The pusher pushes pucks via CONTACT (rich object
dynamics); task = push the TARGET puck into the goal among decoys. This is the scene the whole investigation
pointed to: multi-object, contact, decoys, multimodal planning (which puck, which side to push from).

Validate it's a GENUINE task before any world model:
  (1) a scripted PUSH expert (get behind target puck, push toward goal) reaches the goal,
  (2) a DIRECT/greedy pusher (move toward goal ignoring the puck) FAILS to bring the puck to goal,
  (3) the pusher actually moves pucks via contact (not frictionless / not stuck).
A clean gap (expert works, direct fails) = genuine contact-planning task. Run: MUJOCO_GL=egl python3 -m system1_motion.probe_push
"""
import numpy as np, mujoco

PUSH_XML = """
<mujoco model="pusher2d">
  <option timestep="0.01" gravity="0 0 0" integrator="implicitfast"/>
  <visual><global offwidth="256" offheight="256"/></visual>
  <default>
    <joint type="slide" damping="0.2" limited="true" range="-0.34 0.34"/>
    <geom friction="0.6 0.005 0.0001"/>
  </default>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.95 0.95 0.95"/>
    <camera name="top" pos="0 0 1.05" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="0.4 0.4 0.02" pos="0 0 0" rgba="0.16 0.18 0.22 1"/>
    <body name="goal" mocap="true" pos="0.2 0.2 0.002">
      <geom name="goal" type="cylinder" size="0.055 0.001" rgba="1 0.9 0.25 0.35" contype="0" conaffinity="0"/>
    </body>
    <body name="pusher" pos="-0.2 -0.2 0.03">
      <joint name="px" axis="1 0 0"/><joint name="py" axis="0 1 0"/>
      <geom name="pusher" type="cylinder" size="0.022 0.03" rgba="0.35 0.8 1 1" mass="0.3"/>
    </body>
    <body name="puck0" pos="0.0 0.0 0.03">
      <joint name="p0x" axis="1 0 0" damping="6"/><joint name="p0y" axis="0 1 0" damping="6"/>
      <geom name="puck0" type="cylinder" size="0.038 0.025" rgba="0.3 0.95 0.42 1" mass="0.15"/>
    </body>
    <body name="puck1" pos="0.12 -0.1 0.03">
      <joint name="p1x" axis="1 0 0" damping="6"/><joint name="p1y" axis="0 1 0" damping="6"/>
      <geom name="puck1" type="cylinder" size="0.038 0.025" rgba="0.95 0.35 0.32 1" mass="0.15"/>
    </body>
    <body name="puck2" pos="-0.1 0.12 0.03">
      <joint name="p2x" axis="1 0 0" damping="6"/><joint name="p2y" axis="0 1 0" damping="6"/>
      <geom name="puck2" type="cylinder" size="0.038 0.025" rgba="0.4 0.5 0.95 1" mass="0.15"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="px" kv="12" ctrlrange="-1 1"/><velocity joint="py" kv="12" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class PushEnv:
    def __init__(self, seed=0, ar=4):
        self.m = mujoco.MjModel.from_xml_string(PUSH_XML); self.d = mujoco.MjData(self.m)
        self.ren = mujoco.Renderer(self.m, 96, 96); self.ar = ar; self.rng = np.random.RandomState(seed)
        self.bid = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                    ("pusher", "puck0", "puck1", "puck2")}
        self.PUCK_R, self.PUSH_R = 0.038, 0.022
    def pos(self, name): return self.d.xpos[self.bid[name]][:2].copy()
    def goal(self):      return self.d.mocap_pos[0][:2].copy()
    def set_goal(self, xy): self.d.mocap_pos[0][:2] = xy; mujoco.mj_forward(self.m, self.d)
    def _place(self, name, xy):
        adr = self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)]
        return adr
    def reset(self, layout=None):
        mujoco.mj_resetData(self.m, self.d)
        if layout is None:
            pts = self._sample_layout()
        else:
            pts = layout
        # qpos order: px,py,p0x,p0y,p1x,p1y,p2x,p2y
        self.d.qpos[:] = np.array([pts["pusher"][0], pts["pusher"][1], pts["puck0"][0], pts["puck0"][1],
                                   pts["puck1"][0], pts["puck1"][1], pts["puck2"][0], pts["puck2"][1]])
        self.d.qvel[:] = 0; self.set_goal(pts["goal"]); mujoco.mj_forward(self.m, self.d)
    def _sample_layout(self):
        pts = {}
        names = ["pusher", "puck0", "puck1", "puck2", "goal"]; placed = []
        for n in names:
            for _ in range(200):
                c = self.rng.uniform(-0.28, 0.28, 2)
                if all(np.linalg.norm(c - p) > 0.11 for p in placed): placed.append(c); pts[n] = c; break
            else: pts[n] = self.rng.uniform(-0.28, 0.28, 2); placed.append(pts[n])
        return pts
    def step(self, ctrl):
        self.d.ctrl[:] = np.clip(ctrl, -1, 1)
        for _ in range(self.ar): mujoco.mj_step(self.m, self.d)
    def render(self):
        self.ren.update_scene(self.d, camera="top"); return self.ren.render()
    def move_toward(self, xy, gain=8.0):                          # pusher velocity command toward a point
        v = (np.asarray(xy) - self.pos("pusher")); return np.clip(gain * v, -1, 1)


def expert_push(env, target="puck0", steps=90):
    """ORBIT-then-push: arc the pusher AROUND the puck (staying at contact radius) until it's behind the puck
    (opposite the goal), then push through toward the goal. The orbit avoids knocking the puck the wrong way."""
    R = env.PUCK_R + env.PUSH_R + 0.012
    for t in range(steps):
        p, g, push = env.pos(target), env.goal(), env.pos("pusher")
        to_goal = g - p; d = np.linalg.norm(to_goal)
        if d < 0.04: break
        dirn = to_goal / d; behind_dir = -dirn
        rel = push - p; rel_d = np.linalg.norm(rel) + 1e-9; cur_dir = rel / rel_d
        if np.dot(cur_dir, behind_dir) > 0.6 and rel_d < R + 0.05:    # behind & close -> push through
            env.step(env.move_toward(g, gain=10))
        else:                                                        # orbit around the puck toward the behind point
            tdir = cur_dir + 0.6 * (behind_dir - cur_dir); tdir /= (np.linalg.norm(tdir) + 1e-9)
            env.step(env.move_toward(p + tdir * (R + 0.01), gain=10))
    return float(np.linalg.norm(env.pos(target) - env.goal()))


def direct_push(env, target="puck0", steps=70):
    """greedy baseline: just drive the pusher toward the goal, ignoring the puck."""
    for t in range(steps): env.step(env.move_toward(env.goal()))
    return float(np.linalg.norm(env.pos(target) - env.goal()))


def main():
    env = PushEnv(0)
    print("nq", env.m.nq, "nu", env.m.nu, "render", env.render().shape)
    print("\n=== EXPERT push (behind-then-through) vs DIRECT (greedy to goal), 5 layouts, target=green puck0 ===")
    ex_ok = di_ok = 0; moved = 0
    for i in range(5):
        env.reset(); g0 = np.linalg.norm(env.pos("puck0") - env.goal())
        lay = current_layout(env)                                # same layout for all three methods
        e1 = PushEnv(0); e1.reset(lay); de_e = expert_push(e1)
        e2 = PushEnv(0); e2.reset(lay); de_d = direct_push(e2)
        e3 = PushEnv(0); e3.reset(lay); p_before = e3.pos("puck0").copy(); [e3.step(e3.move_toward(e3.pos("puck0"))) for _ in range(40)]; pmoved = np.linalg.norm(e3.pos("puck0") - p_before)
        ex_ok += de_e < 0.06; di_ok += de_d < 0.06; moved += pmoved > 0.03
        print(f"  layout {i}: start={g0*100:4.1f}cm  EXPERT={de_e*100:4.1f}cm reached={de_e<0.06}   DIRECT={de_d*100:4.1f}cm reached={de_d<0.06}   pusher_moves_puck={pmoved*100:.1f}cm")
    print(f"\n  EXPERT reaches {ex_ok}/5 | DIRECT reaches {di_ok}/5 | pusher-moves-puck {moved}/5")
    print(f"  VERDICT: genuine contact/decoy task IF expert>=4/5 AND direct<=1/5 AND pusher-moves-puck>=4/5")
    print("DONE")


def current_layout(env):
    return {"pusher": env.pos("pusher"), "puck0": env.pos("puck0"), "puck1": env.pos("puck1"),
            "puck2": env.pos("puck2"), "goal": env.goal()}


if __name__ == "__main__":
    main()
