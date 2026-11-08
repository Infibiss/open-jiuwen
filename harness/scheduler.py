"""Budget-aware planner / executor.

Phase 0  zero-token solvers for every task (parse templates, compute, verify).
Phase 1  remaining tasks are priced by the Estimator and ranked by
         reward / expected cost; workers (<= MAX_PARALLEL sessions) walk an
         escalation ladder, reserving the predicted cost before each step:
           oneshot (fast model)  ->  oneshot + error feedback
           ->  oneshot on the reasoning model  ->  ReAct agent (openJiuwen)
Every finished task immediately writes answer.txt and the results file.
"""
import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from . import config, llm_solvers
from .budget import Ledger
from .estimator import Estimator
from .solvers import coding_solver, data_solver, math_solver, search_solver
from .solvers.base import INFEASIBLE, SOLVED
from .tasks import ResultsFile, load_tasks, write_answer

SOLVERS = {"math": math_solver, "data_analysis": data_solver, "search": search_solver, "coding": coding_solver}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


class Harness:
    def __init__(self, tasks_dir, out_path):
        self.t0 = time.time()
        self.tasks = load_tasks(tasks_dir)
        self.ledger = Ledger(config.BUDGET)
        self.est = Estimator()
        self.results = ResultsFile(out_path, self.tasks, self.ledger, self.est)
        self.llm = None
        self.llm_ok = bool(config.LLM_BASE_URL)

    def time_left(self):
        return config.TIME_LIMIT_S - (time.time() - self.t0)

    # ------------------------------------------------------------- results
    def _apply(self, task, res, strategy):
        task.strategy = strategy
        task.reason = res.reason
        if res.status == SOLVED:
            try:
                write_answer(task, res.answer)
                task.status = "solved"
            except OSError as e:
                task.status, task.reason = "error", f"cannot write answer: {e}"
        elif res.status == INFEASIBLE:
            task.status = "infeasible"  # no answer.txt by the rules
        task.t_end = time.time()
        task.cost = self.ledger.task_cost(task.id)
        self.results.save()

    # ------------------------------------------------------------- phase 0
    def _deterministic(self, task):
        task.t_start = time.time()
        solver = SOLVERS.get(task.type)
        if solver is None or config.DISABLE_DETERMINISTIC:
            return task, None
        try:
            return task, solver.solve(task.data, task.cwd)
        except Exception as e:  # noqa: BLE001 - never let one task kill the run
            log(f"{task.id}: solver crash {e!r}")
            return task, None

    def _llm_estimate(self, task):
        """What the task would cost on the LLM ladder (reported for every task)."""
        try:
            prompt = llm_solvers.build(task)
        except Exception:  # noqa: BLE001
            prompt = None
        p = (self.est.predict(task, "oneshot", prompt.text, prompt.target_text) if prompt is not None
             else self.est.predict(task, "react"))
        p["expected_total"] = self.est.expected_total(task, p)
        return p

    def phase0(self):
        # price every task on the LLM ladder first (prompts are built from the untouched workspace)
        with ThreadPoolExecutor(max_workers=4) as ex:
            for task, p in zip(self.tasks, ex.map(self._llm_estimate, self.tasks)):
                task.predicted_llm = p
        with ThreadPoolExecutor(max_workers=4) as ex:
            for task, res in ex.map(self._deterministic, self.tasks):
                if res is not None and res.status in (SOLVED, INFEASIBLE):
                    task.predicted = {"strategy": "deterministic", "cost": 0}
                    self._apply(task, res, "deterministic")
                elif res is not None:
                    task.reason = res.reason
        n_s = sum(t.status == "solved" for t in self.tasks)
        n_i = sum(t.status == "infeasible" for t in self.tasks)
        log(f"phase0: {len(self.tasks)} tasks, solved {n_s}, infeasible {n_i}, "
            f"left {len(self.tasks) - n_s - n_i} ({time.time() - self.t0:.1f}s)")
        self.results.save()

    # ------------------------------------------------------------- phase 1
    async def _step(self, task, strategy, prompt, model, extra_msgs=None):
        """One model call + verification. Returns SolveResult or None (skipped)."""
        messages = list(prompt.messages) + (extra_msgs or [])
        pred = self.est.predict(task, strategy, "\n".join(m["content"] for m in messages), prompt.target_text)
        max_tokens = prompt.max_tokens if strategy == "oneshot" else 4000
        worst = pred["input_tokens"] * config.INPUT_WEIGHT + max_tokens
        reserve = min(worst, max(pred["cost"] * 1.5, pred["cost"] + 200))
        if self.time_left() < 60 or not self.ledger.try_reserve(reserve):
            return None
        before = self.ledger.task_cost(task.id)
        try:
            text, _ = await self.llm.chat(task.id, messages, model=model, max_tokens=max_tokens,
                                          tag=f"{task.type}/{strategy}")
        except Exception as e:  # noqa: BLE001
            log(f"{task.id}: llm error {e}")
            return None
        finally:
            self.ledger.release(reserve)
        res = await asyncio.to_thread(llm_solvers.finish, task, text)
        res.info["reply"] = text[-600:]
        self.est.observe(task, strategy, pred["cost"], self.ledger.task_cost(task.id) - before,
                         res.status == SOLVED)
        return res

    async def _ladder(self, task):
        prompt = None if config.FORCE_REACT else llm_solvers.build(task)
        res = None
        tried_infeasible = False
        if prompt is not None:
            res = await self._step(task, "oneshot", prompt, config.MODEL_FAST)
            if res is not None and res.status == SOLVED:
                return res, "oneshot"
            # one cheap retry with concrete feedback (exec error / failing tests / answer not in KB)
            if res is not None and res.status != INFEASIBLE and res.info.get("reply"):
                fb = [{"role": "assistant", "content": res.info["reply"]},
                      {"role": "user", "content": f"That did not work: {res.reason[:600]}\nFix it. Same output "
                                                  f"format."}]
                res2 = await self._step(task, "oneshot", prompt, config.MODEL_FAST, fb)
                if res2 is not None and res2.status == SOLVED:
                    return res2, "oneshot+feedback"
                res = res2 or res
            tried_infeasible = res is not None and res.status == INFEASIBLE
            if config.ENABLE_REASONING_MODEL and self.ledger.remaining() > 0.15 * self.ledger.usable:
                res3 = await self._step(task, "reason", prompt, config.MODEL_REASON)
                if res3 is not None and res3.status == SOLVED:
                    return res3, "reason"
                if res3 is not None and res3.status == INFEASIBLE and tried_infeasible:
                    return res3, "reason"  # two independent INFEASIBLE verdicts
                if res3 is not None:
                    res = res3
        if config.ENABLE_REACT and self.time_left() > 120 and not self.llm.dead:
            from .react import run_react  # imported lazily: heavier agent stack
            pred = self.est.predict(task, "react")
            cap = max(1500.0, min(pred["cost"] * 2.5, self.ledger.remaining()))
            if self.ledger.try_reserve(min(cap, pred["cost"])):
                self.ledger.release(min(cap, pred["cost"]))
                before = self.ledger.task_cost(task.id)
                try:
                    r = await run_react(task, self.ledger, config.MODEL_FAST, token_cap=before + cap)
                    self.est.observe(task, "react", pred["cost"], self.ledger.task_cost(task.id) - before,
                                     r.status == SOLVED)
                    if r.status in (SOLVED, INFEASIBLE):
                        return r, "react"
                except Exception as e:  # noqa: BLE001
                    log(f"{task.id}: react error {e!r}")
        if tried_infeasible:
            return res, "oneshot"
        return res, "exhausted"

    async def _worker(self, task, sem):
        async with sem:
            if self.time_left() < 60 or self.ledger.remaining() <= 0 or self.llm.dead:
                task.reason = "skipped (budget/time/LLM unavailable)"
                return
            task.t_start = time.time()
            res, strat = await self._ladder(task)
            if res is not None and res.status in (SOLVED, INFEASIBLE):
                self._apply(task, res, strat)
            else:
                task.status, task.strategy = "failed", strat
                task.reason = res.reason if res is not None else "skipped (budget/time)"
                task.cost = self.ledger.task_cost(task.id)
                self.results.save()
            log(f"{task.id}: {task.status} via {task.strategy} answer={task.answer!r} cost={task.cost:.0f} "
                f"pred={task.predicted.get('cost')} spent={self.ledger.spent:.0f}")

    async def phase1(self):
        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            return
        if not self.llm_ok:
            log("no LLM_BASE_URL - skipping LLM phase")
            return
        from .llm import LLM
        self.llm = LLM(self.ledger)
        for t in pending:
            prompt = llm_solvers.build(t)
            if prompt is not None:
                p = self.est.predict(t, "oneshot", prompt.text, prompt.target_text)
            else:
                p = self.est.predict(t, "react")
            p["expected_total"] = self.est.expected_total(t, p)
            t.predicted = p
        pending.sort(key=lambda t: -t.reward / max(1.0, t.predicted["expected_total"]))
        log(f"phase1: {len(pending)} tasks, predicted total {sum(t.predicted['expected_total'] for t in pending):.0f} "
            f"tokens, budget left {self.ledger.remaining():.0f}")
        self.results.save()
        sem = asyncio.Semaphore(config.MAX_PARALLEL)
        await asyncio.gather(*(self._worker(t, sem) for t in pending))

    def predict_only(self):
        """Pre-run forecast without solving: workspaces are copied to /tmp so
        the zero-token solvers can be tried without touching the task files."""
        import shutil
        import tempfile
        for task in self.tasks:
            task.predicted_llm = self._llm_estimate(task)
            tmp = tempfile.mkdtemp(prefix="pred_")
            try:
                ws = task.cwd
                if task.workspace:
                    ws = os.path.join(tmp, "ws")
                    shutil.copytree(task.workspace, ws)
                solver = SOLVERS.get(task.type)
                res = solver.solve(task.data, ws) if solver else None
            except Exception:  # noqa: BLE001
                res = None
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if res is not None and res.status in (SOLVED, INFEASIBLE):
                task.predicted = {"strategy": "deterministic", "cost": 0, "verdict": res.status}
            else:
                task.predicted = dict(task.predicted_llm)
            task.status = "predicted"
        self.results.save()
        total = sum(t.predicted.get("expected_total", t.predicted.get("cost", 0)) for t in self.tasks)
        log(f"predict-only: {len(self.tasks)} tasks, predicted total spend {total:.0f} tokens")

    def run(self):
        self.phase0()
        try:
            asyncio.run(asyncio.wait_for(self.phase1(), timeout=max(30, self.time_left())))
        except asyncio.TimeoutError:
            log("time limit reached")
        self.results.save()
        pts = sum(t.reward for t in self.tasks if t.status == "solved")
        log(f"done: solved {sum(t.status == 'solved' for t in self.tasks)}/{len(self.tasks)}, "
            f"points {pts:g}, tokens {self.ledger.spent:.0f}/{self.ledger.budget:.0f}, "
            f"{time.time() - self.t0:.0f}s")
