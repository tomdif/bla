#!/usr/bin/env python3
"""Web GUI for the drag-the-goal world-model demo. Stdlib HTTP server wrapping TWO DemoEngines (one driven by
the world-model CEM planner, one by imitation BC) that share a single dragged goal, so the contrast renders
SIDE BY SIDE in real time. High-res MuJoCo frames are PNG-compressed; the page is served from wm_demo.html.

  MUJOCO_GL=egl python3 -m system1_motion.wm_demo_server --port 8000
"""
import argparse, base64, io, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import torch
from PIL import Image
from system1_motion.wm_demo import load_wm, load_bc, DemoEngine, CKPT
from system1_motion.r1_imitation_fails import EXTENT

ENG_WM = None   # world-model-driven engine
ENG_BC = None   # imitation-driven engine
HTML = os.path.join(os.path.dirname(__file__), "wm_demo.html")


def png_b64(arr):
    buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(arr, np.uint8)).save(buf, "PNG", compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pack(o):
    o = dict(o); o["frame"] = png_b64(o["frame"]); return o


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store, max-age=0")     # always fetch the latest page
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0)); return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(HTML) as f: self._send(200, f.read(), "text/html")
        else:
            self._send(404, "{}")

    def do_POST(self):
        try:
            if self.path == "/reset":
                wm, bc = ENG_WM.reset(), ENG_BC.reset()
                self._send(200, json.dumps({"wm": pack(wm), "bc": pack(bc)}))
            elif self.path == "/set_goal":
                b = self._body()                                     # {u,v} normalized [0,1] (resolution-independent)
                x = b["u"] * 2 * EXTENT - EXTENT; y = b["v"] * 2 * EXTENT - EXTENT
                g = ENG_WM.set_goal((x, y)); ENG_BC.set_goal((x, y))
                self._send(200, json.dumps(g))
            elif self.path == "/step":
                wm = ENG_WM.step("wm"); bc = ENG_BC.step("bc")
                self._send(200, json.dumps({"wm": pack(wm), "bc": pack(bc)}))
            else:
                self._send(404, "{}")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)}))


def main():
    global ENG_WM, ENG_BC
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--disp", type=int, default=440); args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo-server] loading checkpoints from {CKPT}/ on {dev} ...", flush=True)
    wm = load_wm(f"{CKPT}/wm.pt", dev); bc = load_bc(f"{CKPT}/bc.pt", dev)
    ENG_WM = DemoEngine(wm, bc, dev, seed=0, disp_size=args.disp)         # same seed => identical start states
    ENG_BC = DemoEngine(wm, bc, dev, seed=0, disp_size=args.disp)
    print(f"[demo-server] loaded gated WM (arm_px={wm['arm_px']:.1f}, OOD={wm['rollout_ood_px']:.1f}) + BC", flush=True)
    print("[demo-server] warming up CEM/cuDNN (so the first user action isn't slow) ...", flush=True)
    for _ in range(3): ENG_WM.step("wm"); ENG_BC.step("bc")     # JIT + cuDNN autotune before serving
    ENG_WM.reset(); ENG_BC.reset()
    print(f"[demo-server] warm. serving on 0.0.0.0:{args.port} (disp={args.disp}px) -- open the forwarded URL", flush=True)
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()  # single-threaded: MuJoCo EGL renderer is thread-bound


if __name__ == "__main__":
    main()
