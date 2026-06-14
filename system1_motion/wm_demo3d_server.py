#!/usr/bin/env python3
"""3D drag-the-goal demo on the FetchReach Cartesian world model (runs/r3_ckpt/wm3d.pt) -- the demo-quality model
(reaches shifted 3D goals to ~2cm; moat WM 1.00 vs BC 0.07 @5cm). Stdlib web server (like wm_demo_server.py) wrapping
a FetchReach env: move the 3D goal with x/y/z sliders, toggle World Model vs Imitation, watch the WM reach any goal
while BC fails on SHIFTED goals (y < 0.749, where no demo went). Hi-res scene render; 96px downsample feeds the WM.

  MUJOCO_GL=egl python3 -m system1_motion.wm_demo3d_server --port 8001
"""
import argparse, base64, io, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import torch
from PIL import Image
from system1_motion.models import ViTEncoder, LatentDynamics, DecodeHead
from system1_motion.r3_fetch3d import make_env3d, norm3, cem_plan3d, BC3D, LO, HI, SPAN, Y_SPLIT, ENVID
import gymnasium as gym
try:
    import gymnasium_robotics; gym.register_envs(gymnasium_robotics)
except Exception: pass

ENG = None; HTML = os.path.join(os.path.dirname(__file__), "wm_demo3d.html")
DISP = 380


def load_wm(path, dev):
    ck = torch.load(path, map_location=dev); adim, img = ck["adim"], ck["img"]
    enc = ViTEncoder(img, 8, 3, 384, 6).to(dev); enc.load_state_dict(ck["enc"]); enc.eval()
    dyn = LatentDynamics(384, adim, 4).to(dev); dyn.load_state_dict(ck["dyn"]); dyn.eval()
    dg = DecodeHead(384, out_dim=3).to(dev); dg.load_state_dict(ck["dec_g"]); dg.eval()
    dt = DecodeHead(384, out_dim=3).to(dev); dt.load_state_dict(ck["dec_t"]); dt.eval()
    return {"enc": enc, "dyn": dyn, "dec_g": dg, "dec_t": dt, "adim": adim,
            "grip_cm": ck.get("grip_cm"), "rollout_ood_cm": ck.get("rollout_ood_cm"), "img": img}


class DemoEngine3D:
    def __init__(self, dev):
        self.dev = dev
        self.env = gym.make(ENVID, render_mode="rgb_array", width=DISP, height=DISP)  # hi-res for display
        self.env.reset(seed=0); self.u = self.env.unwrapped
        self.wm = load_wm("runs/r3_ckpt/wm3d.pt", dev)
        ck = torch.load("runs/r3_ckpt/bc3d.pt", map_location=dev)
        self.bc = BC3D(ck["adim"]).to(dev); self.bc.load_state_dict(ck["state"]); self.bc.eval()
        self.goal = np.array([1.34, 0.62, 0.50])              # start with a shifted goal (y<0.749)
        self.set_goal(self.goal)

    def set_goal(self, xyz):
        self.goal = np.clip(np.asarray(xyz, np.float64), LO, HI)
        self.u.goal = self.goal.copy()                        # FetchReach renders the marker at self.goal
        import mujoco; mujoco.mj_forward(self.u.model, self.u.data)
        return {"goal": self.goal.tolist(), "goal_norm": norm3(self.goal).tolist(),
                "region": "test" if self.goal[1] < Y_SPLIT else "train"}

    def reset(self):
        self.env.reset(); self.set_goal(self.goal); return self._obs(self.env.render())

    @torch.no_grad()
    def step(self, method):
        frame = self.env.render()                             # [DISP,DISP,3]
        small = np.asarray(Image.fromarray(frame).resize((self.wm["img"], self.wm["img"]), Image.BILINEAR))
        x = torch.from_numpy(small.transpose(2, 0, 1).astype(np.float32) / 255.0)[None].to(self.dev)
        g = norm3(self.goal)
        if method == "wm":
            z0 = self.wm["enc"](x); a = cem_plan3d(self.wm, z0, g, self.dev)
        elif method == "bc":
            a = self.bc(x, torch.tensor(g, device=self.dev).float()[None]).cpu().numpy()[0]
        else:
            a = np.zeros(self.wm["adim"], np.float32)
        obs, _, term, trunc, _ = self.env.step(np.clip(a, -1, 1).astype(np.float32))
        if term or trunc: self.env.reset(); self.set_goal(self.goal); obs = self.env.step(np.zeros(self.wm["adim"], np.float32))[0]
        return self._obs(self.env.render(), obs)

    def _obs(self, frame, obs=None):
        grip = obs["achieved_goal"].tolist() if obs is not None else self.u.goal.tolist()
        dist = float(np.linalg.norm(np.array(grip) - self.goal)) * 100 if obs is not None else 0.0
        return {"frame": frame, "gripper": grip, "goal": self.goal.tolist(), "dist_cm": dist,
                "region": "test" if self.goal[1] < Y_SPLIT else "train",
                "grip_cm": self.wm.get("grip_cm"), "ood_cm": self.wm.get("rollout_ood_cm"),
                "lo": LO.tolist(), "hi": HI.tolist(), "y_split": Y_SPLIT}


def png_b64(arr):
    buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(arr, np.uint8)).save(buf, "PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def pack(o):
    o = dict(o); o["frame"] = png_b64(o["frame"]); return o


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n = int(self.headers.get("Content-Length", 0)); return json.loads(self.rfile.read(n) or b"{}")
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(HTML) as f: self._send(200, f.read(), "text/html")
        else: self._send(404, "{}")
    def do_POST(self):
        try:
            if self.path == "/reset": self._send(200, json.dumps(pack(ENG.reset())))
            elif self.path == "/set_goal":
                b = self._body(); self._send(200, json.dumps(ENG.set_goal([b["x"], b["y"], b["z"]])))
            elif self.path == "/step":
                b = self._body(); self._send(200, json.dumps(pack(ENG.step(b.get("method", "wm")))))
            else: self._send(404, "{}")
        except Exception as e:
            import traceback; traceback.print_exc(); self._send(500, json.dumps({"error": str(e)}))


def main():
    global ENG
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8001); args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo3d] loading FetchReach Cartesian WM on {dev} ...", flush=True)
    ENG = DemoEngine3D(dev)
    print(f"[demo3d] loaded WM (grip_cm={ENG.wm['grip_cm']:.2f}, OOD={ENG.wm['rollout_ood_cm']:.1f}) + BC", flush=True)
    print(f"[demo3d] serving on 0.0.0.0:{args.port} -- forward the port and open it", flush=True)
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
