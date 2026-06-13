"""WMOS web cockpit -- a local hypothesis dashboard (stdlib http.server, zero deps).

Panels: scene/canvas, typed hypotheses (with status), beliefs, persistent library, audit log,
autonomy control, canary suite. Buttons run governed commands (simulate / verify / act) so a human
operator can watch the verified loop: propose -> imagine -> verify -> act, with the invariant that
no unverified proposal owns truth.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .cli import run_cmd, render_canvas

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>WMOS cockpit</title>
<style>
 body{font:13px ui-monospace,Menlo,monospace;background:#0e1116;color:#d7dee8;margin:0;padding:14px}
 h1{font-size:15px;margin:0 0 10px;color:#7fb3ff}h1 small{color:#6b7686;font-weight:normal}
 .grid{display:grid;grid-template-columns:380px 1fr;gap:12px}
 .panel{background:#161b22;border:1px solid #232a33;border-radius:8px;padding:10px;margin-bottom:12px}
 .panel h2{font-size:12px;margin:0 0 8px;color:#8aa0b8;text-transform:uppercase;letter-spacing:.5px}
 table.canvas{border-collapse:collapse}table.canvas td{width:22px;height:22px;text-align:center}
 .c0{background:#1b2230}.c1{background:#3a4251}.c2{background:#3fb950;color:#06120a;font-weight:bold}
 .c3{background:#58a6ff;color:#04132b;font-weight:bold}.c4{background:#e3b341;color:#1a1402;font-weight:bold}
 .c5{background:#3fb07a;color:#04140c}.cd{background:#1b2230;color:#6b7686}
 .hyp{padding:6px;border-left:3px solid #30363d;margin:4px 0;background:#11161d}
 .needs_measurement{border-color:#e3b341}.verified{border-color:#3fb950}.refuted{border-color:#f85149}
 .ood_refuse{border-color:#a371f7}.trusted{border-color:#58a6ff}
 button{font:12px ui-monospace;background:#21262d;color:#d7dee8;border:1px solid #30363d;border-radius:5px;padding:3px 8px;cursor:pointer;margin:2px}
 button:hover{background:#30363d}.tag{font-size:10px;padding:1px 5px;border-radius:3px;background:#21262d}
 pre{white-space:pre-wrap;margin:0;color:#9fb0c3;font-size:12px}.audit{max-height:160px;overflow:auto}
 .inv{color:#f0883e}.bar{margin-bottom:10px}select{background:#21262d;color:#d7dee8;border:1px solid #30363d;border-radius:5px;padding:2px}
</style></head><body>
<h1>WMOS cockpit <small>&nbsp; invariant: <span class=inv>no unverified proposal owns truth</span></small></h1>
<div class=bar>autonomy <select id=aut onchange="setaut()"><option>manual</option><option>assisted</option><option>auto</option></select>
 <button onclick="cmd('/hypotheses')">re-hypothesize</button>
 <button onclick="cmd('/canaries')">run canaries</button>
 <button onclick="cmd('/reset')">reset</button>
 <span id=msg style=color:#6b7686></span></div>
<div class=grid>
 <div>
  <div class=panel><h2>scene</h2><div id=canvas></div></div>
  <div class=panel><h2>state</h2><pre id=state></pre></div>
  <div class=panel><h2>plain-language</h2><pre id=explain></pre></div>
 </div>
 <div>
  <div class=panel><h2>typed hypotheses (advice; verifier owns truth)</h2><div id=hyps></div></div>
  <div class=panel><h2>beliefs (instance) &amp; library (class)</h2><pre id=beliefs></pre></div>
  <div class=panel><h2>audit log</h2><pre id=audit class=audit></pre></div>
 </div>
</div>
<script>
async function api(p,b){let o={};if(b){o.method='POST';o.body=JSON.stringify(b)}let r=await fetch(p,o);return r.json()}
async function cmd(c){let r=await api('/api/cmd',{cmd:c});document.getElementById('msg').textContent=r.result.split('\\n')[0];refresh()}
async function setaut(){await cmd('/autonomy '+document.getElementById('aut').value)}
function gridhtml(v){if(!v.grid)return v.text||'';let h='<table class=canvas>';for(let r=0;r<v.grid.length;r++){h+='<tr>';for(let c=0;c<v.grid[r].length;c++){let x=v.grid[r][c],cl='c'+x,t={2:'@',3:'G',1:'#',4:'Y',5:'g'}[x]||'';if(v.disguised&&r==v.disguised[0]&&c==v.disguised[1]&&x==0){cl='cd';t='~'}h+='<td class='+cl+'>'+t+'</td>'}h+='</tr>'}return h+'</table>'}
async function refresh(){
 let s=await api('/api/snapshot');
 document.getElementById('aut').value=s.state.autonomy;
 document.getElementById('canvas').innerHTML=gridhtml(s.view);
 document.getElementById('state').textContent='adapter '+s.state.adapter+'  reachable '+s.state.reachable+'  solved '+s.state.solved+'\\n'+s.state.scene;
 document.getElementById('explain').textContent=s.explain;
 document.getElementById('beliefs').textContent=s.beliefs;
 document.getElementById('audit').textContent=s.audit.map(a=>'['+a[0]+'] '+a[1]).reverse().join('\\n')||'(empty)';
 let H='';for(const x of s.hyps){H+='<div class="hyp '+x.status+'"><b>'+x.hid+'</b> <span class=tag>'+x.status+'</span> '+x.label+
  '<br>key='+x.key+' src='+x.source+' conf='+x.confidence+' predΔ='+x.pred_delta+' ood='+x.ood+
  '<br><button onclick="cmd(\\'/simulate '+x.hid+'\\')">simulate</button>'+
  '<button onclick="cmd(\\'/verify '+x.hid+'\\')">verify</button>'+
  '<button onclick="cmd(\\'/act '+x.hid+'\\')">act</button>'+
  '<button onclick="cmd(\\'/why '+x.hid+'\\')">why</button></div>'}
 document.getElementById('hyps').innerHTML=H||'(run re-hypothesize)';
}
cmd('/hypotheses');setInterval(refresh,1500);
</script></body></html>"""


def _snapshot(h):
    if not h.hyps: h.hypothesize()                              # don't re-run proposers (or the LLM) every poll
    obs = h.adapter.observe()
    beliefs = "\n".join(f"{cid} ({v['sig']}) -> {v['effect']}" for cid, v in h.mem.beliefs.items()) or "(none verified)"
    lib = json.dumps(h.mem.library, indent=1) if h.mem.library else "(empty)"
    return {"state": h.state(), "view": obs.get("view", {"text": ""}),
            "explain": run_cmd(h, "/explain"), "beliefs": beliefs + "\n--- library ---\n" + lib,
            "audit": h.mem.audit[-40:], "hyps": [x.to_dict() for x in h.hyps.values()]}


def make_handler(h):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            try:
                if self.path == "/": return self._send(200, PAGE, "text/html; charset=utf-8")
                if self.path == "/api/snapshot": return self._send(200, json.dumps(_snapshot(h)))
                if self.path == "/api/state": return self._send(200, json.dumps(h.state()))
                self._send(404, json.dumps({"error": "not found"}))
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/cmd":
                    return self._send(200, json.dumps({"result": run_cmd(h, body.get("cmd", "/help"))}))
                self._send(404, json.dumps({"error": "not found"}))
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
    return Handler


def serve(h, host="127.0.0.1", port=8765):
    srv = ThreadingHTTPServer((host, port), make_handler(h))
    print(f"WMOS cockpit -> http://{host}:{port}   (adapter={h.adapter.name}, Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping cockpit."); srv.shutdown()
    return 0
