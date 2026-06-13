"""WMOS test suite -- the consumer-grade guarantees, including the central invariant."""
import os, tempfile, unittest
from wmos import Harness, get_adapter, list_adapters, SessionStore, load_config
from wmos.cli import run_cmd


def fresh(adapter="grid"):
    store = SessionStore(tempfile.mkdtemp())                    # isolated library/sessions per test
    return Harness(get_adapter(adapter), store, autonomy="manual")


class TestInvariant(unittest.TestCase):
    def test_unverified_action_is_blocked(self):
        h = fresh(); h.hypothesize()
        hid = next(iter(h.hyps))
        r = h.act(hid)
        self.assertFalse(r["released"], "shadow + invariant: unverified proposal must not act")

    def test_verification_owns_truth(self):
        h = fresh(); h.hypothesize()
        # the real switch verifies positive; the trap (identical signature) is refuted
        deltas = {hid: h.verify(hid).measured_delta for hid in list(h.hyps)}
        self.assertIn(37.0, deltas.values(), "switch should measure Δ>0")
        self.assertIn(0.0, deltas.values(), "trap should measure Δ=0 and be refuted")

    def test_verified_action_releases(self):
        h = fresh(); h.hypothesize()
        sw = next(hid for hid in h.hyps if h.adapter.measure_delta(h.hyps[hid].cid) > 0)
        h.verify(sw)
        self.assertTrue(h.act(sw)["released"])
        self.assertTrue(h.adapter.observe()["solved"])

    def test_refuted_action_blocked(self):
        h = fresh(); h.hypothesize()
        tr = next(hid for hid in h.hyps if h.adapter.measure_delta(h.hyps[hid].cid) == 0)
        h.verify(tr)
        self.assertFalse(h.act(tr)["released"], "a refuted proposal must not act")


class TestGovernor(unittest.TestCase):
    def test_ood_is_refused(self):
        h = fresh()
        _p, _b, ood = h.est.predict({"adj_wall": 1, "dist": 999})
        self.assertTrue(ood, "estimator must flag out-of-distribution candidates")

    def test_autonomy_dial_gates(self):
        h = fresh(); h.autonomy = "auto"; h.hypothesize()
        sw = next(hid for hid in h.hyps if h.adapter.measure_delta(h.hyps[hid].cid) > 0)
        # under 'auto' a high-confidence in-band proposal can release without explicit verify
        self.assertEqual(h.hyps[sw].status, "trusted")


class TestPersistence(unittest.TestCase):
    def test_library_persists_across_harnesses(self):
        base = tempfile.mkdtemp()
        h1 = Harness(get_adapter("grid"), SessionStore(base)); h1.hypothesize()
        sw = next(hid for hid in h1.hyps if h1.adapter.measure_delta(h1.hyps[hid].cid) > 0)
        h1.verify(sw)
        h2 = Harness(get_adapter("grid"), SessionStore(base))   # new harness, same storage
        self.assertTrue(h2.mem.library, "verified affordances must persist across sessions")

    def test_session_report_round_trip(self):
        store = SessionStore(tempfile.mkdtemp())
        h = Harness(get_adapter("grid"), store); h.hypothesize()
        path = store.save_session(h.session_id, h.report())
        self.assertTrue(os.path.exists(path))
        self.assertIn(h.session_id, store.list_sessions())


class TestAdapters(unittest.TestCase):
    def test_registry(self):
        self.assertIn("grid", list_adapters()); self.assertIn("reach", list_adapters())

    def test_reach_adapter_runs(self):
        h = fresh("reach"); h.hypothesize()
        self.assertTrue(h.hyps, "reach adapter must yield candidates")
        tool = next((hid for hid in h.hyps if h.adapter.measure_delta(h.hyps[hid].cid) > 0), None)
        self.assertIsNotNone(tool, "grabbing a tool must increase achievable targets")


class TestARCAdapter(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa
        except Exception:
            self.skipTest("numpy required for the ARC adapter")

    def test_arc_perceives_and_verifier_owns_truth(self):
        from wmos.adapters.arc import ARCAdapter, SyntheticLs20Source
        a = ARCAdapter(source=SyntheticLs20Source())
        obs = a.observe()
        self.assertGreater(obs["reachable"], 5, "should flood the maze corridor")
        self.assertTrue(any("cross" in c["id"] for c in obs["candidates"]), "should perceive the cross operator")
        h = Harness(a, SessionStore(tempfile.mkdtemp())); h.hypothesize()
        self.assertTrue(h.hyps, "WMOS must produce hypotheses on real-format ls20 frames")
        # the cross is an OPERATOR (shape-flip), not a reachability gate -> a reachability verifier
        # correctly REFUTES it. The verifier owning truth (no false-positive affordance) is the point.
        hid = next(iter(h.hyps))
        self.assertEqual(h.verify(hid).status, "refuted")

    def test_arc_registered(self):
        from wmos import list_adapters
        self.assertIn("arc", list_adapters())


class TestLs20Hierarchy(unittest.TestCase):
    def test_richer_signal_confirms_cross_where_flat_refutes(self):
        a = get_adapter("ls20")
        self.assertGreater(a.measure_delta("cross"), 0, "richer (shape) signal must advance the win")
        self.assertEqual(a.flat_reachability_delta("cross"), 0.0, "flat reachability is blind -> would refute")

    def test_decoy_refuted(self):
        self.assertEqual(get_adapter("ls20").measure_delta("yellow"), 0.0)

    def test_wmos_verifies_cross_on_ls20(self):
        a = get_adapter("ls20"); h = Harness(a, SessionStore(tempfile.mkdtemp())); h.hypothesize()
        cross = next(hid for hid, x in h.hyps.items() if x.cid == "cross")
        self.assertEqual(h.verify(cross).status, "verified", "the richer signal makes the cross a real affordance")

    def test_hierarchy_ordering(self):
        a = get_adapter("ls20"); g = a.goals()
        self.assertEqual(g["frontier"], "shape_matched")
        at_exit = next(s for s in g["subgoals"] if s["name"] == "at_exit")
        self.assertFalse(at_exit["ready"], "at_exit must require shape_matched first (ordering)")

    def test_full_solve_respects_ordering(self):
        a = get_adapter("ls20")
        a.go_to_exit()                                   # try to reach exit BEFORE matching
        self.assertFalse(a.observe()["solved"], "reaching the exit unmatched must NOT win (the gate)")
        a.flip_to_match(); a.go_to_exit()
        self.assertTrue(a.observe()["solved"], "match-then-exit wins")
        self.assertEqual(a.goals()["frontier"], None)

    def test_goals_command(self):
        h = Harness(get_adapter("ls20"), SessionStore(tempfile.mkdtemp()))
        out = run_cmd(h, "/goals")
        self.assertIn("shape_matched", out); self.assertIn("at_exit", out)


class TestCLI(unittest.TestCase):
    def test_commands_never_crash(self):
        h = fresh()
        for c in ["/state", "/canvas", "/hypotheses", "/why H1", "/simulate H1", "/verify H1",
                  "/act H1", "/canaries", "/library", "/explain", "/autonomy auto", "/bogus", "/why", "/adapters"]:
            out = run_cmd(h, c)
            self.assertIsInstance(out, str)
            self.assertNotIn("Traceback", out)

    def test_config_defaults_and_env(self):
        cfg = load_config("/nonexistent/path.json")
        self.assertEqual(cfg["adapter"], "grid")
        os.environ["WMOS_AUTONOMY"] = "auto"
        self.assertEqual(load_config("/nonexistent/path.json")["autonomy"], "auto")
        del os.environ["WMOS_AUTONOMY"]


if __name__ == "__main__":
    unittest.main()
