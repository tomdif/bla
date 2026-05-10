"""Router action enum + RouterAction dataclass.

The B.L.A. router's action space is the *budget allocator's* decision:
at each reasoning step, what does the system spend compute on?

  ANSWER     emit the current best response
  RETRIEVE   query memory (vector / symbolic / episodic / executable)
  SIMULATE   run the world model forward under candidate actions
  PROVE      invoke the verification layer on a sub-claim
  SEARCH     counterexample search against the current draft
  ASK        emit a clarifying query to the user / environment
  DEFER      escalate to a longer-running tool / human / batch

Phase 2 supplies the API + the dispatch table. Learned routing is Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RouterActionType(Enum):
    ANSWER = "answer"
    RETRIEVE = "retrieve"
    SIMULATE = "simulate"
    PROVE = "prove"
    SEARCH = "search"
    ASK = "ask"
    DEFER = "defer"


@dataclass
class RouterAction:
    type: RouterActionType
    budget_flops: int = 0
    payload: Any = None
    details: dict = field(default_factory=dict)
