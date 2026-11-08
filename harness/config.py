"""Runtime configuration (environment variables + defaults)."""
import os


def _int(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# Budget in *output-equivalent* tokens: cost = input/10 + output (+ reasoning).
BUDGET = _int("BUDGET", 100_000)
INPUT_WEIGHT = _float("INPUT_TOKEN_WEIGHT", 0.1)
# Never plan to spend the last few percent: our usage accounting may differ
# slightly from the proxy's, and an exhausted proxy kills the run.
SAFETY_FRACTION = _float("BUDGET_SAFETY_FRACTION", 0.06)

# Models (qwen3.8-27b is the stronger dense model; -noreason = no thinking tokens)
MODEL_FAST = os.environ.get("MODEL_FAST", "qwen3.8-27b-noreason")
MODEL_ALT = os.environ.get("MODEL_ALT", "qwen3.6-35b-a3b-noreason")
MODEL_REASON = os.environ.get("MODEL_REASON", "qwen3.8-27b")

MAX_PARALLEL = max(1, min(5, _int("MAX_PARALLEL", 4)))
TIME_LIMIT_S = _int("TIME_LIMIT_S", 55 * 60)  # container is killed at 60 min
LLM_TIMEOUT_S = _int("LLM_TIMEOUT_S", 120)
REQUEST_RETRIES = _int("LLM_RETRIES", 2)

# Escalation ladder switches
ENABLE_REACT = os.environ.get("ENABLE_REACT", "1") != "0"
ENABLE_REASONING_MODEL = os.environ.get("ENABLE_REASONING_MODEL", "1") != "0"
REACT_MAX_ITERS = _int("REACT_MAX_ITERS", 8)

# Write a copy of answer.txt into workspace/ as well (instructions say "current
# directory"; organisers said "next to task.json" - we do both).
ANSWER_IN_WORKSPACE = os.environ.get("ANSWER_IN_WORKSPACE", "1") != "0"

# Testing / trajectory collection: skip the zero-token solvers so every task
# goes through the LLM ladder.
DISABLE_DETERMINISTIC = os.environ.get("DISABLE_DETERMINISTIC", "0") == "1"
# Testing: jump straight to the ReAct agent (exercises the openJiuwen agent loop).
FORCE_REACT = os.environ.get("FORCE_REACT", "0") == "1"
