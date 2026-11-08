"""Common result type for zero-token (deterministic) solvers."""
from dataclasses import dataclass, field
from typing import Optional

SOLVED = "solved"          # answer computed, verified by construction
INFEASIBLE = "infeasible"  # task provably cannot be answered from given data
UNKNOWN = "unknown"        # template not recognised -> escalate to LLM


@dataclass
class SolveResult:
    status: str
    answer: Optional[str] = None
    reason: str = ""
    # free-form extra info for the trajectory log
    info: dict = field(default_factory=dict)

    @classmethod
    def solved(cls, answer, reason="", **info):
        return cls(SOLVED, str(answer), reason, info)

    @classmethod
    def infeasible(cls, reason, **info):
        return cls(INFEASIBLE, None, reason, info)

    @classmethod
    def unknown(cls, reason, **info):
        return cls(UNKNOWN, None, reason, info)
