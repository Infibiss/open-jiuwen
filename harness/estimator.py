"""Pre-run token-cost prediction (in output-equivalent tokens).

cost(task, strategy) = calls * (input_tokens * 0.1 + output_tokens)

* input_tokens is known almost exactly before the call: we build the prompt
  first and measure it (chars / 3.6).
* output_tokens comes from per-(task type, strategy) priors fitted on our
  collected trajectories (see README / trajectories/), scaled for coding by
  the size of the file the model must rewrite.
* An online multiplicative correction (EMA of actual / predicted) per
  (type, strategy) adapts the priors during the run.

The scheduler uses `expected_total` (cost of the cheap attempt + failure
probability * cost of the escalation) to rank tasks by reward / cost and to
reserve budget before starting a task.
"""
import threading

from . import config
from .budget import approx_tokens

# Priors fitted on trajectories/ (qwen3.8-27b, 87 calls): mean output tokens
# per call, calls per attempt, success probability of the attempt.
PRIORS = {
    ("math", "oneshot"): (65, 1.0, 0.92),
    ("data_analysis", "oneshot"): (95, 1.0, 0.90),
    ("search", "oneshot"): (33, 1.0, 0.92),
    ("coding", "oneshot"): (None, 1.1, 0.85),  # None -> size of the file the model re-emits
    ("default", "oneshot"): (200, 1.0, 0.6),
    ("any", "reason"): (260, 1.0, 0.8),  # thinking tokens included
    ("any", "react"): (55, 3.5, 0.75),
}
# chars per prompt token measured on our prompts (Qwen tokenizer; names / digits tokenize densely)
CHARS_PER_TOKEN = {"math": 3.95, "search": 2.95, "data_analysis": 3.2, "coding": 3.25}
TEMPLATE_OVERHEAD = 20  # chat-template tokens per request
REACT_CONTEXT = 950  # mean prompt tokens per ReAct iteration (system + tool schema + history)


class Estimator:
    def __init__(self):
        self.scale = {}
        self.stats = {}  # key -> [n, success]
        self._lock = threading.Lock()

    def _prior(self, ttype, strategy):
        if strategy == "oneshot":
            return PRIORS.get((ttype, "oneshot"), PRIORS[("default", "oneshot")])
        return PRIORS[("any", strategy)]

    def predict(self, task, strategy, prompt_text="", target_text=""):
        out, calls, _ = self._prior(task.type, strategy)
        cpt = CHARS_PER_TOKEN.get(task.type, 3.3)
        tin = int(len(prompt_text) / cpt) + TEMPLATE_OVERHEAD if prompt_text else 900
        if strategy == "react":
            tin = REACT_CONTEXT + approx_tokens(task.instruction)
        if out is None:  # coding: the model re-emits the whole file
            out = int(len(target_text) / cpt) + 15
        raw = calls * (tin * config.INPUT_WEIGHT + out)
        with self._lock:
            s = self.scale.get((task.type, strategy), 1.0)
        return {"strategy": strategy, "input_tokens": int(tin * calls), "output_tokens": int(out * calls),
                "cost": round(raw * s, 1)}

    def p_success(self, task, strategy):
        _, _, p0 = self._prior(task.type, strategy)
        with self._lock:
            n, k = self.stats.get((task.type, strategy), (0, 0))
        return (p0 * 4 + k) / (4 + n)  # Beta prior with strength 4

    def expected_total(self, task, first):
        """Expected spend to finish `task` starting from prediction `first`."""
        cost = first["cost"]
        p = self.p_success(task, first["strategy"])
        esc = self.predict(task, "reason")["cost"] if config.ENABLE_REASONING_MODEL else 0
        return round(cost + (1 - p) * esc, 1)

    def observe(self, task, strategy, predicted, actual, success):
        key = (task.type, strategy)
        with self._lock:
            if predicted > 0 and actual > 0:
                r = min(4.0, max(0.25, actual / predicted))
                self.scale[key] = 0.7 * self.scale.get(key, 1.0) + 0.3 * r
            n, k = self.stats.get(key, (0, 0))
            self.stats[key] = (n + 1, k + (1 if success else 0))

    def state(self):
        with self._lock:
            return {"scale": {f"{a}/{b}": round(v, 3) for (a, b), v in self.scale.items()},
                    "stats": {f"{a}/{b}": v for (a, b), v in self.stats.items()}}
