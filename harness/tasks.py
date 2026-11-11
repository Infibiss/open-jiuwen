"""Task discovery, answer writing and the results file."""
import json
import os
import threading
import time

from . import config
from .fsutil import atomic_write


class Task:
    def __init__(self, task_dir, data):
        self.dir = task_dir
        self.data = data
        self.id = data.get("task_id") or os.path.basename(task_dir)
        self.type = data.get("task_type") or self.id.rsplit("_", 1)[0]
        self.instruction = data.get("instruction", "")
        try:
            self.reward = float(data.get("reward", 1))
        except (TypeError, ValueError):
            self.reward = 1.0
        ws = os.path.join(task_dir, "workspace")
        self.workspace = ws if os.path.isdir(ws) else None
        # bookkeeping filled by the scheduler
        self.status = "pending"
        self.answer = None
        self.strategy = None
        self.reason = ""
        self.predicted = {}      # plan actually used (0 for zero-token solvers)
        self.predicted_llm = {}  # what the LLM ladder would cost (reported for every task)
        self.det_info = {}       # what the zero-token solver learned (e.g. ambiguous repairs)
        self.cost = 0.0
        self.t_start = None
        self.t_end = None
        self.laddered = False    # went up the LLM escalation ladder (phase 1)
        self.tried = []          # ladder rungs already run ("strategy/model", "retry:<rung>")
        self.prompt = None       # one-shot prompt of the ladder (reused by the second pass)
        self.last_reply = ""     # last failed model reply (feedback for the second pass)

    @property
    def cwd(self):
        """The agent's 'current directory' for this task."""
        return self.workspace or self.dir

    def to_json(self):
        return {"task_id": self.id, "task_type": self.type, "reward": self.reward, "status": self.status,
                "answer": self.answer, "strategy": self.strategy, "reason": self.reason[:300],
                "predicted_cost": self.predicted, "predicted_llm_cost": self.predicted_llm,
                "actual_cost": round(self.cost, 1),
                "seconds": round((self.t_end or time.time()) - self.t_start, 2) if self.t_start else None}


def load_tasks(tasks_dir):
    tasks = []
    for name in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, name)
        tj = os.path.join(d, "task.json")
        if not os.path.isfile(tj):
            continue
        try:
            with open(tj, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        tasks.append(Task(d, data))
    return tasks


def write_answer(task, answer):
    text = str(answer).strip() + "\n"
    paths = [os.path.join(task.dir, "answer.txt")]
    if config.ANSWER_IN_WORKSPACE and task.workspace:
        paths.append(os.path.join(task.workspace, "answer.txt"))
    for i, p in enumerate(paths):
        try:
            atomic_write(p, text)
        except OSError:
            if i == 0:
                raise  # the official answer file could not be written
    task.answer = str(answer).strip()


def remove_answer(task):
    for p in (os.path.join(task.dir, "answer.txt"),
              os.path.join(task.workspace, "answer.txt") if task.workspace else None):
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


class ResultsFile:
    """Solutions file, rewritten atomically after every finished task."""

    def __init__(self, path, tasks, ledger, estimator=None):
        self.path = path
        self.tasks = tasks
        self.ledger = ledger
        self.estimator = estimator
        self._lock = threading.Lock()
        self.t0 = time.time()

    def save(self):
        if not self.path:
            return
        with self._lock:
            payload = {
                "elapsed_s": round(time.time() - self.t0, 1),
                "ledger": self.ledger.snapshot(),
                "solved": sum(t.status == "solved" for t in self.tasks),
                "infeasible": sum(t.status == "infeasible" for t in self.tasks),
                "expected_points": sum(t.reward for t in self.tasks if t.status == "solved"),
                "answers": {t.id: t.answer for t in self.tasks if t.status == "solved"},
                "tasks": [t.to_json() for t in self.tasks],
                "calls": self.ledger.calls,
            }
            if self.estimator is not None:
                payload["estimator"] = self.estimator.state()
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            except OSError:
                pass


def resolve_out_path(arg_out):
    out = arg_out or os.environ.get("OUT") or "/tmp/results.json"
    if out.endswith(os.sep) or os.path.isdir(out):
        out = os.path.join(out, "results.json")
    return out
