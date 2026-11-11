"""Budget-aware planner / executor.

Phase 0  zero-token solvers for every task (parse templates, compute, verify).
Phase 1  remaining tasks are priced by the Estimator and ranked by reward / expected
         cost. Workers (<= MAX_PARALLEL sessions) reserve the predicted cost before
         each step and walk the ladder in passes, so cheap attempts on every task
         come before expensive retries on any single task:
           pass 1  oneshot (fast model) -> oneshot + error feedback
                   (an INFEASIBLE verdict is checked once by a second cheap model)
           pass 2  failed tasks, best value first: reasoning model -> ReAct agent
           pass 3  only while plenty of budget is left: ReAct on the reasoning model
                   for tasks still failing, reasoning-model recheck of LLM INFEASIBLE
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
from .solvers.base import INFEASIBLE, SOLVED, SolveResult
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
                    task.det_info = res.info
        n_s = sum(t.status == "solved" for t in self.tasks)
        n_i = sum(t.status == "infeasible" for t in self.tasks)
        log(f"phase0: {len(self.tasks)} tasks, solved {n_s}, infeasible {n_i}, "
            f"left {len(self.tasks) - n_s - n_i} ({time.time() - self.t0:.1f}s)")
        self.results.save()

    # ------------------------------------------------------------- phase 1
    async def _reserve(self, amount):
        """Reserve budget for a step. While other workers hold reservations, wait for
        them to settle instead of giving up: their real cost is usually far below what
        they hold. Fail only when the budget cannot fit the step even with nothing held."""
        while True:
            if self.ledger.try_reserve(amount):
                return True
            if self.ledger.reserved <= 0 or self.ledger.usable - self.ledger.spent < amount \
                    or self.time_left() < 60:
                return False
            await asyncio.sleep(0.2)

    async def _step(self, task, strategy, prompt, model, extra_msgs=None):
        """One model call + verification. Returns SolveResult or None (skipped)."""
        messages = list(prompt.messages) + (extra_msgs or [])
        pred = self.est.predict(task, strategy, "\n".join(m["content"] for m in messages), prompt.target_text)
        max_tokens = prompt.max_tokens if strategy == "oneshot" else 4000
        worst = pred["input_tokens"] * config.INPUT_WEIGHT + max_tokens
        # generous margin: a waiting worker loses only time, an overrun breaks the budget
        reserve = min(worst, max(pred["cost"] * 1.5, pred["cost"] + 200))
        if self.time_left() < 60 or not await self._reserve(reserve):
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
        try:
            res = await asyncio.to_thread(llm_solvers.finish, task, text)
        except Exception as e:  # noqa: BLE001 - e.g. PermissionError while applying a patch
            log(f"{task.id}: verification error {e!r}")
            res = SolveResult.unknown(f"verification error: {e!r}"[:300])
        res.info["reply"] = text[-600:]
        self.est.observe(task, strategy, pred["cost"], self.ledger.task_cost(task.id) - before,
                         res.status == SOLVED)
        return res

    @staticmethod
    def _feedback(res):
        """Previous reply + what was wrong with it, for a retry."""
        if res is None or not res.info.get("reply"):
            return None
        return [{"role": "assistant", "content": res.info["reply"]},
                {"role": "user", "content": f"That did not work: {res.reason[:600]}\nFix it. Same output format."}]

    async def _pass1(self, task):
        """Cheap attempts only: fast model, one retry with feedback, second cheap
        opinion on INFEASIBLE. Returns (result, strategy)."""
        prompt = None if config.FORCE_REACT else llm_solvers.build(task)
        if prompt is None:
            return None, "no-prompt"
        res = await self._step(task, "oneshot", prompt, config.MODEL_FAST)
        if res is not None and res.status == SOLVED:
            return res, "oneshot"
        # one cheap retry with concrete feedback (exec error / failing tests / answer not in KB)
        if res is not None and res.status != INFEASIBLE:
            fb = self._feedback(res)
            res2 = await self._step(task, "oneshot", prompt, config.MODEL_FAST, fb) if fb else None
            if res2 is not None and res2.status == SOLVED:
                return res2, "oneshot+feedback"
            if res.info.get("tradeoff") and res2 is not None and res2.info.get("tradeoff"):
                # twice: fixing the failing tests broke a passing one -> contradictory tests
                return SolveResult.infeasible("tests contradict each other (every fix breaks another test)"), \
                    "oneshot+feedback"
            res = res2 or res
        if res is not None and res.status == INFEASIBLE:
            # cheap independent second opinion (other noreason model) instead of the
            # reasoning model: an infeasible task earns nothing, so spend as little as possible
            alt = await self._step(task, "oneshot", prompt, config.MODEL_ALT)
            if alt is not None and alt.status == SOLVED:
                return alt, "oneshot-alt"
            return res, "oneshot+alt-confirm"
        return res, "oneshot"

    async def _react(self, task, model, cap):
        """openJiuwen ReAct agent, stopped by force_finish once the task has spent `cap` more."""
        from .react import run_react  # imported lazily: heavier agent stack
        pred = self.est.predict(task, "react")
        hold = min(cap, max(pred["cost"], cap if model == config.MODEL_REASON else 0))
        if self.time_left() < 120 or self.llm.dead or not await self._reserve(hold):
            return None
        before = self.ledger.task_cost(task.id)
        try:
            r = await run_react(task, self.ledger, model, token_cap=before + cap)
            self.est.observe(task, "react", pred["cost"], self.ledger.task_cost(task.id) - before,
                             r.status == SOLVED)
            return r
        except Exception as e:  # noqa: BLE001
            log(f"{task.id}: react error {e!r}")
            return None
        finally:
            self.ledger.release(hold)

    async def _pass2(self, task):
        """Escalation for a task the cheap pass could not close."""
        prompt = None if config.FORCE_REACT else llm_solvers.build(task)
        res = task.last_result
        fb = self._feedback(res)
        if prompt is not None and fb:
            # the other cheap model with the last failure: about the price of one fast call,
            # often enough when the fast model kept repeating its own mistake
            r = await self._step(task, "oneshot", prompt, config.MODEL_ALT, fb)
            if r is not None and r.status == SOLVED:
                return r, "alt+feedback"
            res = r if r is not None and r.status != INFEASIBLE else res
        if prompt is not None and config.ENABLE_REASONING_MODEL:
            r = await self._step(task, "reason", prompt, config.MODEL_REASON, self._feedback(res))
            if r is not None and r.status in (SOLVED, INFEASIBLE):
                return r, "reason"
            res = r or res
        if config.ENABLE_REACT:
            pred = self.est.predict(task, "react")
            r = await self._react(task, config.MODEL_FAST, max(1500.0, min(pred["cost"] * 2.5, self.ledger.remaining())))
            if r is not None and r.status in (SOLVED, INFEASIBLE):
                return r, "react"
            res = r or res
        return res, "exhausted"

    async def _pass3(self, task):
        """Leftover budget only: the strongest (and most expensive) attempts."""
        if task.status == "infeasible":  # an LLM verdict: let the reasoning model look once more
            prompt = llm_solvers.build(task)
            r = await self._step(task, "reason", prompt, config.MODEL_REASON) if prompt is not None else None
            return (r, "pass3-recheck") if r is not None and r.status == SOLVED else (None, None)
        if config.ENABLE_REACT:
            r = await self._react(task, config.MODEL_REASON, min(6000.0, self.ledger.remaining()))
            if r is not None and r.status in (SOLVED, INFEASIBLE):
                return r, "pass3-react"
        return None, None

    async def _worker(self, task, sem, stage):
        async with sem:
            if self.time_left() < 60 or self.ledger.remaining() <= 0 or self.llm.dead:
                if stage == 1:
                    task.reason = "skipped (budget/time/LLM unavailable)"
                return
            task.t_start = task.t_start or time.time()
            try:
                res, strat = await {1: self._pass1, 2: self._pass2, 3: self._pass3}[stage](task)
                if res is not None and res.status in (SOLVED, INFEASIBLE):
                    if not (stage == 3 and res.status == INFEASIBLE):
                        self._apply(task, res, strat)
                elif stage < 3:
                    task.status, task.strategy = "failed", strat
                    task.reason = res.reason if res is not None else "skipped (budget/time)"
                    task.last_result = res or task.last_result
            except Exception as e:  # noqa: BLE001 - one broken task must not stop the others
                if stage < 3:
                    task.status, task.reason = "failed", f"worker error: {e!r}"[:300]
                log(f"{task.id}: worker error {e!r}")
            task.cost = self.ledger.task_cost(task.id)
            self.results.save()
            log(f"{task.id}: pass{stage} {task.status} via {task.strategy} answer={task.answer!r} "
                f"cost={task.cost:.0f} spent={self.ledger.spent:.0f}")

    async def _run_pass(self, stage, tasks, key):
        if not tasks or self.llm.dead or self.time_left() < 120 or self.ledger.remaining() <= 0:
            return
        tasks = sorted(tasks, key=key)
        log(f"pass{stage}: {len(tasks)} tasks, budget left {self.ledger.remaining():.0f}")
        sem = asyncio.Semaphore(config.MAX_PARALLEL)
        await asyncio.gather(*(self._worker(t, sem, stage) for t in tasks), return_exceptions=True)

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
        log(f"phase1: {len(pending)} tasks, predicted total {sum(t.predicted['expected_total'] for t in pending):.0f} "
            f"tokens, budget left {self.ledger.remaining():.0f}")
        self.results.save()
        value = lambda t: -t.reward / max(1.0, t.predicted["expected_total"])  # noqa: E731
        await self._run_pass(1, pending, value)
        failed = [t for t in pending if t.status == "failed"]
        reason_cost = lambda t: -t.reward / max(1.0, self.est.predict(t, "reason")["cost"])  # noqa: E731
        await self._run_pass(2, failed, reason_cost)
        # pass 3 spends what is left over; it never starts while the budget is tight
        if self.ledger.remaining() > config.PASS3_MIN_FRACTION * self.ledger.usable:
            still = [t for t in pending if t.status == "failed"]
            llm_inf = [t for t in pending if t.status == "infeasible" and t.strategy == "oneshot+alt-confirm"]
            await self._run_pass(3, still, value)
            if self.ledger.remaining() > config.PASS3_MIN_FRACTION * self.ledger.usable:
                await self._run_pass(3, llm_inf, value)

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
        except Exception as e:  # noqa: BLE001 - answers already written stay valid
            log(f"LLM phase aborted: {e!r}")
        self.results.save()
        pts = sum(t.reward for t in self.tasks if t.status == "solved")
        log(f"done: solved {sum(t.status == 'solved' for t in self.tasks)}/{len(self.tasks)}, "
            f"points {pts:g}, tokens {self.ledger.spent:.0f}/{self.ledger.budget:.0f}, "
            f"{time.time() - self.t0:.0f}s")
