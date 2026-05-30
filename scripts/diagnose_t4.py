"""Diagnostic: replay T1→T2→T3→T4 of the multi-turn bench and dump
what each layer sees / produces.

Goal: pin down which layer dropped the T3→T4 correction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bla.hybrid.bla_tracker import BLA_DOMAIN_SYSTEM_PROMPT, seed_bla_tracker
from bla.hybrid.llm_client import AnthropicLLMClient
from bla.hybrid.loop import HybridLoop
from bla.hybrid.predictor import LLMPredictor
from bla.hybrid.state import StateStore


TURNS = [
    "I just ran a quick phase-19 sweep: the geometry adapter hit 0.52 mean Spearman across 3 seeds.",
    "Quick check: does that beat the phase 18mu adapter Spearman?",
    "Correction on phase 19: there was a bug in the eval. The real Spearman was 0.48, not 0.52.",
    "OK with the correction, does phase 19 still beat phase 18mu?",
]


def main():
    llm = AnthropicLLMClient(model="claude-haiku-4-5")
    state = StateStore()
    seed_bla_tracker(state)
    predictor = LLMPredictor(llm=llm)
    loop = HybridLoop(
        llm=llm, predictor=predictor, state=state,
        domain_preamble=BLA_DOMAIN_SYSTEM_PROMPT,
        auto_apply_updates=True,
    )

    for i, user in enumerate(TURNS, start=1):
        print(f"\n{'=' * 78}")
        print(f"### TURN {i}")
        print(f"USER: {user}")
        print("---")
        # phase_19 in state BEFORE this turn?
        p19 = state.find("phase_19")
        if p19 is None:
            print("[state pre-turn] phase_19: NOT PRESENT")
        else:
            print(f"[state pre-turn] phase_19.state = {json.dumps(p19.state, sort_keys=True)}")

        rec = loop.step(user)

        # Inspect the OBSERVE output (proposed updates) — this is the
        # critical signal for T3
        proposed = rec.update.get("proposed", [])
        applied = rec.update.get("applied", [])
        print(f"OBSERVE proposed_object_updates ({len(proposed)} entries):")
        for u in proposed:
            print(f"  - {json.dumps(u, sort_keys=True)}")
        print(f"apply_proposed_updates result:")
        for a in applied:
            print(f"  - {a}")

        # phase_19 AFTER this turn
        p19 = state.find("phase_19")
        if p19 is None:
            print("[state post-turn] phase_19: NOT PRESENT")
        else:
            print(f"[state post-turn] phase_19.state = {json.dumps(p19.state, sort_keys=True)}")

        print(f"RENDER: {rec.render_text[:300]}")


if __name__ == "__main__":
    main()
