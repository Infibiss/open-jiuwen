"""Token ledger: the single source of truth for budget decisions.

The proxy does not report the remaining budget, so every LLM call made by the
harness (single-shot calls and every ReAct iteration) is charged here from the
`usage` block of the response:

    cost = prompt_tokens * 0.1 + completion_tokens (+ reasoning tokens when the
           provider reports them outside completion_tokens)
"""
import threading
import time

from . import config


def approx_tokens(text):
    """Cheap tokenizer-free estimate (Qwen BPE ~3.6 chars/token on this data)."""
    return int(len(text) / 3.6) + 4


def call_cost(prompt_tokens, completion_tokens, reasoning_tokens=0, reasoning_included=True):
    out = completion_tokens or 0
    if reasoning_tokens and not reasoning_included:
        out += reasoning_tokens
    return (prompt_tokens or 0) * config.INPUT_WEIGHT + out


class Ledger:
    def __init__(self, budget):
        self.budget = float(budget)
        self.spent = 0.0
        self.reserved = 0.0
        self.calls = []
        self._lock = threading.Lock()

    # -- accounting -----------------------------------------------------
    def charge(self, task_id, model, prompt_tokens, completion_tokens, reasoning_tokens=0,
               reasoning_included=True, tag=""):
        c = call_cost(prompt_tokens, completion_tokens, reasoning_tokens, reasoning_included)
        with self._lock:
            self.spent += c
            self.calls.append({"t": round(time.time(), 2), "task": task_id, "model": model, "tag": tag,
                               "in": prompt_tokens, "out": completion_tokens, "reason": reasoning_tokens,
                               "cost": round(c, 1)})
        return c

    def task_cost(self, task_id):
        with self._lock:
            return sum(c["cost"] for c in self.calls if c["task"] == task_id)

    # -- planning -------------------------------------------------------
    @property
    def usable(self):
        return self.budget * (1.0 - config.SAFETY_FRACTION)

    def remaining(self):
        with self._lock:
            return self.usable - self.spent - self.reserved

    def try_reserve(self, amount):
        """Reserve an estimated amount before starting a task (so parallel
        workers cannot jointly overshoot)."""
        with self._lock:
            if self.usable - self.spent - self.reserved >= amount:
                self.reserved += amount
                return True
            return False

    def release(self, amount):
        with self._lock:
            self.reserved = max(0.0, self.reserved - amount)

    def snapshot(self):
        with self._lock:
            return {"budget": self.budget, "spent": round(self.spent, 1), "reserved": round(self.reserved, 1),
                    "calls": len(self.calls)}
