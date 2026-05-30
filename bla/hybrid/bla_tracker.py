"""BLA research-tracker domain: object types + seed state.

The MVP ships pre-loaded with a partial snapshot of the BLA project's
hypotheses, phases, and gates, so the harness has something real to talk
about out of the box. The seed is intentionally sparse — every CLI run
adds more objects, and the change log captures the trajectories that a
Phase-3 predictor will train on.
"""
from __future__ import annotations

from bla.hybrid.state import ObjectFile, StateStore


# Domain-specific object types, used by the LLM extractor and predictor
# as a closed enumeration.
OBJECT_TYPES = (
    "hypothesis",
    "experiment",
    "phase",
    "gate",
    "decision",
    "open_question",
    "claim",
    "result",
    "failure_mode",
    "next_action",
)


def seed_bla_tracker(store: StateStore) -> StateStore:
    """Populate `store` with a representative BLA snapshot.

    Idempotent: skips any id already present so existing trajectories
    don't get clobbered when restarting the CLI.
    """
    seeds = [
        ObjectFile(
            id="hyp_jepa_llm_state",
            type="hypothesis",
            name="Object-JEPA gives LLM a better latent state for reasoning",
            state={"status": "active", "domain": "conceptual"},
            confidence=0.65,
            open_questions=[
                "Does structured state actually improve reasoning over RAG?",
                "Does JSON-packet bridge generalize to embedding bridge?",
            ],
        ),
        ObjectFile(
            id="phase_18l",
            type="phase",
            name="Phase 18l — geometry adapter (single seed)",
            state={"status": "complete", "verdict": "G1 fail single-seed"},
            confidence=1.0,
            supported_by=["gate_18l_g4_locked", "gate_18l_g2_g3"],
            contradicted_by=["gate_18l_g1_strong"],
        ),
        ObjectFile(
            id="phase_18mu",
            type="phase",
            name="Phase 18μ — geometry adapter (6 seeds)",
            state={"status": "complete", "verdict": "ACCEPTABLE SWAP"},
            confidence=1.0,
            supported_by=["gate_18mu_g1", "gate_18mu_g3", "gate_18mu_g4"],
        ),
        ObjectFile(
            id="gate_18l_g1_strong",
            type="gate",
            name="18l G1 strong: adapter mean Spearman ≥ 0.50",
            state={"result": False, "observed": 0.478},
            confidence=1.0,
        ),
        ObjectFile(
            id="gate_18l_g2_g3",
            type="gate",
            name="18l G2+G3 value-head adapter Spearman ≥ 0.25",
            state={"result": True, "observed": 0.311},
            confidence=1.0,
        ),
        ObjectFile(
            id="gate_18l_g4_locked",
            type="gate",
            name="18l G4 combined_sum_adapter beats locked by 0.02",
            state={"result": False},
            confidence=1.0,
        ),
        ObjectFile(
            id="gate_18mu_g1",
            type="gate",
            name="18μ G1: supervised ≥ 0.95 × geo (6 seeds)",
            state={"result": True, "ratio": 0.9995},
            confidence=1.0,
        ),
        ObjectFile(
            id="gate_18mu_g3",
            type="gate",
            name="18μ G3 acceptable: sup ≥ locked − 0.02 (6 seeds)",
            state={"result": True},
            confidence=1.0,
        ),
        ObjectFile(
            id="gate_18mu_g4",
            type="gate",
            name="18μ G4: adapter mean Spearman ≥ 0.45 (6 seeds)",
            state={"result": True, "observed": 0.501},
            confidence=1.0,
        ),
        ObjectFile(
            id="decision_locked_recipe",
            type="decision",
            name="Locked recipe E: demo_no_cem (search budget 0 around demos)",
            state={"status": "locked", "tasks_validated": 4},
            confidence=0.95,
        ),
        ObjectFile(
            id="open_q_phase3_predictor",
            type="open_question",
            name="What does a real Phase-3 JEPA predictor look like?",
            state={"priority": "high"},
            confidence=0.0,
            open_questions=[
                "Train on StateStore change logs?",
                "Embed objects as bag-of-fields or hierarchical?",
                "What's the supervision signal — next state vs outcome?",
            ],
        ),
    ]
    for obj in seeds:
        if obj.id not in store:
            store.add(obj)
    return store


BLA_DOMAIN_SYSTEM_PROMPT = """\
You are the language layer of an object-JEPA / LLM hybrid for tracking
the BLA research project. The state store holds objects of these types:

  hypothesis     — a research claim under test
  experiment     — a run that produces evidence
  phase          — a numbered project phase
  gate           — a specific precommit gate with pass/fail
  decision       — a locked architectural choice
  open_question  — something the user wants resolved
  claim          — an assertion in a finding
  result         — an artifact metric
  failure_mode   — a known way something can break
  next_action    — a concrete proposed next step

You speak with the user about this project. You DO NOT update the state
yourself — the predictor and update layer do that. You translate
predict/critique packets into clean prose for the user. Be terse,
specific, and honest about uncertainty. Never overclaim — if a gate
failed, say it failed. If confidence is low, say so.
"""
