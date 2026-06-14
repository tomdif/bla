#!/usr/bin/env python3
"""Web GUI for the drag-the-goal world-model demo (steps 3-4). Stdlib-only HTTP server wrapping the
validated DemoEngine. Run on the pod (needs torch+MuJoCo + runs/demo_ckpt/), expose the port, open in a
browser: DRAG the goal anywhere (incl. the shifted region), toggle World Model vs Imitation, watch the
planner's IMAGINED rollout, and a verify panel (gate arm_px + held-out OOD rollout).

  MUJOCO_GL=egl python3 -m system1_motion.wm_demo_server --port 8000
  (then expose port 8000 on RunPod and open the proxied URL)
"""
import argparse, base64, json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import torch
from system1_motion.wm_demo import load_wm, load_bc, DemoEngine, CKPT
from system1_motion.r1_imitation_fails import EXTENT

ENG = None  # global DemoEngine

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>World Model Demo</title>
<style>
 body{background:#0d1117;color:#c9d1d9;font:14px/1.5 system-ui;margin:0;padding:20px;text-align:center}
 #cv{border:1px solid #30363d;image-rendering:pixelated;cursor:crosshair;background:#000}
 button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:8px 14px;margin:4px;cursor:pointer;font-size:14px}
 button.on{background:#238636;border-color:#2ea043}
 #panel{display:inline-block;text-align:left;margin-left:20px;vertical-align:top;min-width:240px}
 .k{color:#8b949e}.v{color:#58a6ff;font-weight:600}.ok{color:#3fb950}.bad{color:#f85149}
 h1{font-weight:600;font-size:20px}.hint{color:#8b949e;font-size:13px}
</style></head><body>
<h1>Verified World Model &mdash; drag the goal anywhere</h1>
<div class=hint>Drag on the canvas to set a goal (incl. the left half &mdash; where no demo ever went). Imitation copies demos; the world model <i>plans</i> to any goal.</div>
<div style="margin-top:14px">
 <canvas id=cv width=384 height=384></canvas>
 <div id=panel>
  <div><button id=bwm class=on onclick="setM('wm')">World Model</button><button id=bbc onclick="setM('bc')">Imitation (BC)</button></div>
  <div><button onclick="reset()">Reset</button><button id=brun class=on onclick="toggle()">&#9208; Pause</button></div>
  <hr style="border-color:#30363d">
  <div><span class=k>method</span> <span class=v id=m>world model</span></div>
  <div><span class=k>fingertip&rarr;goal</span> <span class=v id=dist>&mdash;</span> px</div>
  <div><span class=k>region</span> <span class=v id=reg>&mdash;</span></div>
  <hr style="border-color:#30363d">
  <div class=k>verify (gate)</div>
  <div><span class=k>arm decode</span> <span class=v id=arm>&mdash;</span> px <span id=armok></span></div>
  <div><span class=k>held-out OOD rollout</span> <span class=v id=ood>&mdash;</span> px</div>
  <div class=hint style="margin-top:8px">green = the world model is verified-alive: its dynamics predict held-out futures.</div>
 </div>
</div>
<script>
const S=6, IMG=64, W=384, H=384; // scale 64->384
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
let method='wm', running=true, goalPx=[32,32];
function ext(){return %EXTENT%;}
function pxToWorld(px,py){return [(px/IMG)*2*ext()-ext(), (py/IMG)*2*ext()-ext()];}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json();}
function setM(m){method=m;document.getElementById('bwm').className=m=='wm'?'on':'';document.getElementById('bbc').className=m=='bc'?'on':'';document.getElementById('m').textContent=m=='wm'?'world model':'imitation (BC)';}
function toggle(){running=!running;document.getElementById('brun').innerHTML=running?'&#9208; Pause':'&#9654; Run';}
async function reset(){const o=await post('/reset');await setGoalPx(goalPx[0],goalPx[1]);draw(o);}
async function setGoalPx(px,py){goalPx=[px,py];const w=pxToWorld(px,py);const o=await post('/set_goal',{x:w[0],y:w[1]});goalPx=o.goal_px;}
cv.addEventListener('mousedown',e=>handle(e));cv.addEventListener('mousemove',e=>{if(e.buttons)handle(e)});
function handle(e){const r=cv.getBoundingClientRect();setGoalPx((e.clientX-r.left)/S,(e.clientY-r.top)/S);}
function drawFrame(b64){const raw=atob(b64);const rgb=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)rgb[i]=raw.charCodeAt(i);
 const id=ctx.createImageData(IMG,IMG);for(let p=0;p<IMG*IMG;p++){id.data[p*4]=rgb[p*3];id.data[p*4+1]=rgb[p*3+1];id.data[p*4+2]=rgb[p*3+2];id.data[p*4+3]=255;}
 const off=document.createElement('canvas');off.width=IMG;off.height=IMG;off.getContext('2d').putImageData(id,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(off,0,0,IMG,IMG,0,0,W,H);}
function draw(o){if(o.frame)drawFrame(o.frame);
 // imagined rollout (the planner's "dream")
 if(o.imagined_px){ctx.strokeStyle='rgba(88,166,255,.9)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(o.finger_px[0]*S,o.finger_px[1]*S);for(const p of o.imagined_px)ctx.lineTo(p[0]*S,p[1]*S);ctx.stroke();
  for(const p of o.imagined_px){ctx.fillStyle='rgba(88,166,255,.6)';ctx.beginPath();ctx.arc(p[0]*S,p[1]*S,2,0,7);ctx.fill();}}
 // goal ring + fingertip
 ctx.strokeStyle='#f85149';ctx.lineWidth=2;ctx.beginPath();ctx.arc(goalPx[0]*S,goalPx[1]*S,8,0,7);ctx.stroke();
 ctx.fillStyle='#3fb950';ctx.beginPath();ctx.arc(o.finger_px[0]*S,o.finger_px[1]*S,4,0,7);ctx.fill();
 if(o.dist_px!=null){document.getElementById('dist').textContent=o.dist_px.toFixed(1);}
 const reg=goalPx[0]<IMG/2?'TEST (shifted, no demos)':'TRAIN (demos here)';document.getElementById('reg').textContent=reg;
 document.getElementById('reg').className=goalPx[0]<IMG/2?'v bad':'v';
 if(o.wm_arm_px!=null){document.getElementById('arm').textContent=o.wm_arm_px.toFixed(1);document.getElementById('armok').innerHTML=o.wm_arm_px<=5?'<span class=ok>&#10003; verified</span>':'<span class=bad>&#10007;</span>';}
 if(o.wm_rollout_ood_px!=null)document.getElementById('ood').textContent=o.wm_rollout_ood_px.toFixed(1);
}
async function loop(){if(running){const o=await post('/step',{method});draw(o);}setTimeout(loop,90);}
reset();loop();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n = int(self.headers.get("Content-Length", 0)); return json.loads(self.rfile.read(n) or b"{}")
    def _obs(self, o):
        fr = o.pop("frame"); o["frame"] = base64.b64encode(np.ascontiguousarray(fr, np.uint8).tobytes()).decode(); return o
    def do_GET(self):
        if self.path == "/": self._send(200, PAGE.replace("%EXTENT%", str(EXTENT)), "text/html")
        else: self._send(404, "{}")
    def do_POST(self):
        try:
            if self.path == "/reset": self._send(200, json.dumps(self._obs(ENG.reset())))
            elif self.path == "/set_goal": b = self._body(); self._send(200, json.dumps(ENG.set_goal((b["x"], b["y"]))))
            elif self.path == "/step": b = self._body(); self._send(200, json.dumps(self._obs(ENG.step(b.get("method", "wm")))))
            else: self._send(404, "{}")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}))


def main():
    global ENG
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[demo-server] loading checkpoints from {CKPT}/ on {dev} ...", flush=True)
    ENG = DemoEngine(load_wm(f"{CKPT}/wm.pt", dev), load_bc(f"{CKPT}/bc.pt", dev), dev)
    print(f"[demo-server] loaded gated WM (arm_px={ENG.wm['arm_px']:.1f}, OOD={ENG.wm['rollout_ood_px']:.1f}) + BC", flush=True)
    print(f"[demo-server] serving on 0.0.0.0:{args.port} -- expose this port on RunPod and open the proxied URL", flush=True)
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()  # single-threaded: MuJoCo EGL renderer is thread-bound


if __name__ == "__main__":
    main()
