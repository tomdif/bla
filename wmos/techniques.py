"""Technique Discovery layer -- the researcher/experimenter loop, one level above WMOS affordances.

An ACTION changes the world (move, toggle, patch, apply a rewrite). A TECHNIQUE is a reusable
STRATEGY for resolving a class of bottleneck ("when one-step probes plateau, test 2-3 step micro-plans
before rejecting"). The system does NOT get credit for a plausible-sounding technique -- only for one
that SURVIVES A VERIFIER (environment feedback / Δachievable / tests / Lean / held-out). Verified
techniques are PROMOTED to a cumulative library keyed by their bottleneck preconditions, so the next
problem with the same signature reuses the card instead of re-discovering it.

Same invariant, lifted: techniques are PROPOSED (language / library / mutation / search); the world's
truth OWNS promotion. No technique enters the library without surviving its falsification test.
"""
from dataclasses import dataclass, field


@dataclass
class TechniqueHypothesis:
    name: str
    source: str                      # language | library | generator | mutation | analogy
    applies_when: dict               # bottleneck signature it claims to resolve
    predicted_gain: float = 0.0
    falsification_test: str = "apply in the environment; promote iff it resolves the bottleneck"


@dataclass
class EvidenceBundle:
    predicted: dict
    observed: dict
    verdict: str                     # promote | reject


@dataclass
class TechniqueCard:
    name: str
    preconditions: dict
    verified_effects: list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    confidence: float = 0.0


def matches(preconditions, signature):
    """A card/technique applies when ALL its preconditions are present (and equal) in the signature."""
    return bool(preconditions) and all(signature.get(k) == v for k, v in preconditions.items())


class TechniqueLibrary:
    def __init__(self): self.cards = {}
    def propose(self, signature):
        return [c for c in self.cards.values() if matches(c.preconditions, signature)]
    def promote(self, name, preconditions, evidence, effect=None):
        c = self.cards.get(name) or TechniqueCard(name, dict(preconditions))
        c.preconditions = dict(preconditions); c.evidence.append(evidence)
        if effect: c.verified_effects.append(effect)
        c.confidence = round(1 - 0.5 ** len(c.evidence), 3)
        self.cards[name] = c; return c
    def shuffle(self):               # adversarial control: scramble which card claims which precondition
        names = list(self.cards)
        precs = [dict(self.cards[n].preconditions) for n in names][::-1]
        for n, p in zip(names, precs): self.cards[n].preconditions = p
    def inspect(self):
        return [f"{c.name}: applies_when {c.preconditions} (conf {c.confidence}, evidence {len(c.evidence)})"
                for c in self.cards.values()]
