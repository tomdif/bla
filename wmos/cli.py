"""WMOS operator console -- robust REPL + scriptable CLI.

  wmos                         # interactive console (grid adapter, manual/shadow autonomy)
  wmos --adapter reach         # point at a different world
  wmos --demo                  # run the scripted cockpit demo
  wmos --serve                 # launch the web cockpit
"""
import argparse, sys, json
from .adapters import get_adapter, list_adapters
from .engine import Harness
from .persistence import SessionStore
from .config import load_config

HELP = """commands:
  /state                show world + agent state
  /canvas               render the scene (grid adapters)
  /hypotheses           run proposers -> typed hypotheses on the bus (advice only)
  /why <id>             trust ledger (provenance chain) for a hypothesis
  /simulate <id>        imagine the outcome (world model rollout; no action)
  /verify <id>          measure Δachievable (verification owns truth)
  /act <id>             release the action (gated by governor + autonomy)
  /autonomy <lvl>       manual | assisted | auto
  /canaries             run falsifiers (make cheating obvious)
  /library              the persistent affordance library (inspectable)
  /explain              plain-language summary of current beliefs
  /report               export a session audit
  /reset                restart the episode
  /adapters             list available world adapters
  /help                 this help
  /quit                 exit"""

_GRIDSYM = {0: ".", 1: "#", 2: "@", 3: "G", 4: "Y", 5: "g"}


def render_canvas(h):
    obs = h.adapter.observe(); view = obs.get("view", {})
    if "grid" in view:
        dis = view.get("disguised")
        rows = []
        for r, row in enumerate(view["grid"]):
            s = ""
            for c, v in enumerate(row):
                ch = _GRIDSYM.get(v, "?")
                if (r, c) == dis and v == 0: ch = "~"
                s += ch
            rows.append("   " + s)
        return "\n".join(rows) + "\n   legend: @ agent  G goal  # wall  Y yellow  g green  ~ disguised(solid)  . floor"
    if "reach" in view:
        return f"   reach radius: {view['reach']}  targets at radii {view['targets']}  goal radius {view['goal_r']}"
    return "   (no visual view for this adapter)"


def run_cmd(h, line):
    try:
        parts = line.split(); c = parts[0]; arg = parts[1] if len(parts) > 1 else None
        if c in ("/help", "?"): return HELP
        if c == "/adapters": return "available adapters: " + ", ".join(list_adapters())
        if c == "/reset": h.adapter.reset(); h.hyps = {}; return "episode reset."
        if c == "/autonomy":
            if arg in h.AUTONOMY: h.autonomy = arg
            elif arg: return f"unknown level '{arg}'. use: manual | assisted | auto"
            return f"autonomy = {h.autonomy}  (manual=shadow | assisted=trust high-confidence | auto=self-act)"
        if c == "/state":
            s = h.state()
            return (f"adapter={s['adapter']} autonomy={s['autonomy']} | reachable={s['reachable']} solved={s['solved']}\n"
                    f"beliefs={s['beliefs']} library={s['library']} contested={s['contested']}\n  {s['scene']}")
        if c == "/canvas": return render_canvas(h)
        if c == "/hypotheses":
            h.hypothesize()
            if not h.hyps: return "no candidate affordances perceived."
            return "typed hypotheses (proposers = ADVICE; verifier owns truth):\n" + "\n".join(
                f"  {x.hid}  key={x.key}  src={x.source}  conf={x.confidence}  predΔ={x.pred_delta} band={x.band} "
                f"ood={x.ood}  -> {x.status.upper()}" for x in h.hyps.values())
        if c == "/why":
            if not arg or arg not in h.hyps: return f"usage: /why <id>  (active: {list(h.hyps)})"
            x = h.hyps[arg]
            return f"trust ledger {arg} (key={x.key}):\n" + "\n".join(f"   [{t}] {w}" for t, w in x.provenance)
        if c == "/simulate":
            if not arg or arg not in h.hyps: return f"usage: /simulate <id>  (active: {list(h.hyps)})"
            r = h.simulate(arg)
            return (f"IMAGINE {arg}: predicted reachable -> {r['predicted_reachable']} (Δ {r['imagined_delta']:+.0f}); "
                    f"estimator predΔ {r['estimator_pred']} band {r['band']}; status {r['status']}  (no action released)")
        if c == "/verify":
            if not arg or arg not in h.hyps: return f"usage: /verify <id>  (active: {list(h.hyps)})"
            x = h.verify(arg)
            return (f"VERIFY {arg}: measured Δachievable = {x.measured_delta:+.0f} -> {x.status.upper()}\n"
                    f"   belief: {x.key} @ {x.cid} -> {'switch' if x.measured_delta > 0 else 'inert'}")
        if c == "/act":
            if not arg or arg not in h.hyps: return f"usage: /act <id>  (active: {list(h.hyps)})"
            r = h.act(arg)
            return (f"RELEASED action on {r['cid']} (solved={r['solved']})" if r["released"]
                    else f"BLOCKED: {r['reason']}")
        if c == "/canaries":
            return "canary suite:\n" + "\n".join(f"  [{r}] {n} -- {w}" for n, r, w in h.canaries())
        if c == "/library":
            return "persistent affordance library:\n" + (json.dumps(h.mem.library, indent=2) if h.mem.library else "  (empty)")
        if c == "/explain":
            bel = [f"{cid} ({v['sig']}) is a {v['effect']}" for cid, v in h.mem.beliefs.items()]
            con = h.mem.contested()
            note = (f"\n   note: signature {con} is CONTESTED (same look, different behavior) -- needs a finer key."
                    if con else "")
            return ("I propose what each object might do, but only BELIEVE it after testing whether it changes what "
                    "I can reach. I never act on an untested guess unless you raise autonomy.\n   beliefs: "
                    + ("; ".join(bel) if bel else "nothing verified yet") + note)
        if c == "/report":
            path = h.mem.store.save_session(h.session_id, h.report()) if h.mem.store else None
            return f"exported session audit -> {path}" if path else "no store configured."
        return f"unknown command: {line}  (try /help)"
    except Exception as e:                                       # consumer-grade: never crash the console
        return f"error: {type(e).__name__}: {e}"


DEMO = ["/state", "/canvas", "/hypotheses", "/why H1", "/simulate H1", "/act H1",
        "/verify H1", "/act H1", "/why H2", "/verify H2", "/canaries", "/library", "/explain", "/report"]


def make_harness(cfg):
    store = SessionStore(cfg["storage"])
    return Harness(get_adapter(cfg["adapter"]), store, autonomy=cfg["autonomy"],
                   model=cfg["model"], trust_threshold=cfg["trust_threshold"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wmos", description="World-Model Operating System operator console")
    ap.add_argument("--adapter", help="world adapter (default from config: grid)")
    ap.add_argument("--autonomy", choices=["manual", "assisted", "auto"])
    ap.add_argument("--demo", action="store_true", help="run the scripted cockpit demo")
    ap.add_argument("--serve", action="store_true", help="launch the web cockpit")
    ap.add_argument("--config", help="path to a config json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.adapter: cfg["adapter"] = args.adapter
    if args.autonomy: cfg["autonomy"] = args.autonomy
    try:
        h = make_harness(cfg)
    except KeyError as e:
        print(e); return 2
    if args.serve:
        from .server import serve
        return serve(h, cfg["host"], cfg["port"])
    if args.demo:
        print("=" * 74); print("WMOS scripted cockpit demo  --  invariant: NO UNVERIFIED PROPOSAL OWNS TRUTH"); print("=" * 74)
        for line in DEMO:
            print(f"\nwmos> {line}"); print(run_cmd(h, line))
        return 0
    print(f"WMOS v{__import__('wmos').__version__}  adapter={cfg['adapter']}  autonomy={cfg['autonomy']}")
    print("operator console. /help for commands, /quit to exit.")
    while True:
        try:
            line = input("\nwmos> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if line in ("/quit", "/exit"): break
        if line: print(run_cmd(h, line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
