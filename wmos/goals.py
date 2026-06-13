"""Hierarchical sub-goals. A goal decomposes into ordered sub-goals; the achievable signal becomes a
hierarchical POTENTIAL over them, and the FRONTIER (the next unmet sub-goal whose prerequisites are
satisfied) tells the verifier which progress an affordance must advance.

This is what lets WMOS handle compositional wins (e.g. ls20: match the key-shape via the cross, THEN
reach the exit) where flat reachability is blind. Domain-agnostic: an adapter supplies the predicates;
this module orchestrates ordering, frontier, value, and achievement.
"""
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class SubGoal:
    name: str
    satisfied: Callable          # state -> bool
    progress: Callable           # state -> float in [0,1]
    requires: List[str] = field(default_factory=list)   # prerequisite sub-goal names (ordering)
    weight: float = 1.0


class GoalHierarchy:
    def __init__(self, subgoals, root_satisfied):
        self.subgoals = {sg.name: sg for sg in subgoals}
        self.order = [sg.name for sg in subgoals]
        self.root_satisfied = root_satisfied            # state -> bool  (the WIN predicate)

    def ready(self, name, state):
        return all(self.subgoals[r].satisfied(state) for r in self.subgoals[name].requires)

    def achieved(self, state):
        return bool(self.root_satisfied(state))

    def frontier(self, state):
        """next unsatisfied sub-goal whose prerequisites are met -- the one to work on now."""
        for name in self.order:
            sg = self.subgoals[name]
            if not sg.satisfied(state) and self.ready(name, state):
                return sg
        return None

    def value(self, state):
        """hierarchical potential: completed sub-goals (weighted) + progress on the ready frontier."""
        v = 0.0
        for name in self.order:
            sg = self.subgoals[name]
            if sg.satisfied(state):
                v += sg.weight
            elif self.ready(name, state):
                v += sg.weight * max(0.0, min(1.0, sg.progress(state)))
        return v

    def snapshot(self, state):
        front = self.frontier(state)
        return {"achieved": self.achieved(state), "value": round(self.value(state), 3),
                "frontier": front.name if front else None,
                "subgoals": [{"name": n, "satisfied": self.subgoals[n].satisfied(state),
                              "progress": round(self.subgoals[n].progress(state), 3),
                              "ready": self.ready(n, state), "requires": self.subgoals[n].requires}
                             for n in self.order]}
