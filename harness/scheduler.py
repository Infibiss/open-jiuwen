"""Budget-aware planner / executor.

Phase 0  zero-token solvers for every task (parse templates, compute, verify).
Phase 1  remaining tasks are priced by the Estimator and ranked by
         reward / expected cost; workers (<= MAX_PARALLEL sessions) walk an
         escalation ladder, reserving the predicted cost before each step:
           oneshot (fast model)  ->  oneshot + error feedback
           ->  oneshot on the reasoning model  ->  ReAct agent (openJiuwen)
Phase 2  budget left once every task has had its ladder goes back to the tasks
         that failed (or were skipped while other workers held reservations):
         rounds of one untried rung per task, best reward per expected token
         first - another model with the last failure as feedback, the reasoning
         model, the ReAct agent, sampled retries - until no rung fits the budget.
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
    async def _step(self, task, strategy, prompt, model, extra_msgs=None, temperature=0.0, key=None):
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
            task.tried.append(key or f"{strategy}/{model}")
            text, _ = await self.llm.chat(task.id, messages, model=model, max_tokens=max_tokens,
                                          temperature=temperature, tag=f"{task.type}/{strategy}")
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
        if res.status != SOLVED:
            task.last_reply, task.reason = text[-600:], res.reason
        self.est.observe(task, strategy, pred["cost"], self.ledger.task_cost(task.id) - before,
                         res.status == SOLVED)
        return res

    async def _ladder(self, task):
        prompt = None if config.FORCE_REACT else llm_solvers.build(task)
        task.prompt = prompt
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
                if res.info.get("tradeoff") and res2 is not None and res2.info.get("tradeoff"):
                    # twice: fixing the failing tests broke a passing one -> contradictory tests
                    return SolveResult.infeasible("tests contradict each other (every fix breaks another test)"), \
                        "oneshot+feedback"
                res = res2 or res
            tried_infeasible = res is not None and res.status == INFEASIBLE
            if tried_infeasible:
                # cheap independent second opinion (other noreason model) instead of the
                # reasoning model: an infeasible task earns nothing, so spend as little as possible
                alt = await self._step(task, "oneshot", prompt, config.MODEL_ALT)
                if alt is not None and alt.status == SOLVED:
                    return alt, "oneshot-alt"
                return res, "oneshot+alt-confirm"
            if config.ENABLE_REASONING_MODEL and self.ledger.remaining() > 0.15 * self.ledger.usable:
                res3 = await self._step(task, "reason", prompt, config.MODEL_REASON)
                if res3 is not None and res3.status == SOLVED:
                    return res3, "reason"
                if res3 is not None and res3.status == INFEASIBLE:
                    return res3, "reason"
                if res3 is not None:
                    res = res3
        if config.ENABLE_REACT and self.time_left() > 120 and not self.llm.dead:
            r = await self._react(task, config.MODEL_FAST)
            if r is not None and r.status in (SOLVED, INFEASIBLE):
                return r, "react"
            res = r or res
        return res, "exhausted"

    async def _react(self, task, model, key=None):
        """The openJiuwen ReAct agent under a token cap. Returns SolveResult or None (not started)."""
        from .react import run_react  # imported lazily: heavier agent stack
        pred = self.est.predict(task, "react")
        cap = max(1500.0, min(pred["cost"] * 2.5, self.ledger.remaining()))
        hold = min(cap, pred["cost"])
        if not self.ledger.try_reserve(hold):
            return None
        task.tried.append(key or f"react/{model}")
        before = self.ledger.task_cost(task.id)
        try:
            r = await run_react(task, self.ledger, model, token_cap=before + cap)
            self.est.observe(task, "react", pred["cost"], self.ledger.task_cost(task.id) - before,
                             r.status == SOLVED)
            if r.status not in (SOLVED, INFEASIBLE):
                task.reason = r.reason
            return r
        except Exception as e:  # noqa: BLE001
            log(f"{task.id}: react error {e!r}")
            return None
        finally:
            self.ledger.release(hold)

    async def _worker(self, task, sem):
        async with sem:
            if self.time_left() < 60 or self.ledger.remaining() <= 0 or self.llm.dead:
                task.reason = "skipped (budget/time/LLM unavailable)"
                return
            task.laddered = True
            task.t_start = time.time()
            try:
                res, strat = await self._ladder(task)
                if res is not None and res.status in (SOLVED, INFEASIBLE):
                    self._apply(task, res, strat)
                else:
                    task.status, task.strategy = "failed", strat
                    task.reason = res.reason if res is not None else "skipped (budget/time)"
            except Exception as e:  # noqa: BLE001 - one broken task must not stop the others
                task.status, task.reason = "failed", f"worker error: {e!r}"[:300]
                log(f"{task.id}: worker error {e!r}")
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
        await asyncio.gather(*(self._worker(t, sem) for t in pending), return_exceptions=True)

    # ------------------------------------------------------------- phase 2
    def _next_rung(self, task):
        """Cheapest untried way to attack a task that failed its ladder: (key, kind, model,
        temperature, strategy, feedback). None when everything was tried."""
        prompt = getattr(task, "prompt", None)
        rungs = []
        if prompt is not None:
            rungs.append(("alt+fb", "oneshot", config.MODEL_ALT, 0.0, "oneshot", True))
            if config.ENABLE_REASONING_MODEL:
                rungs.append(("reason+fb", "oneshot", config.MODEL_REASON, 0.0, "reason", True))
        if config.ENABLE_REACT:
            rungs.append(("react", "react", config.MODEL_FAST, 0.0, "react", False))
        if prompt is not None:
            rungs += [(f"sample{i}", "oneshot", config.MODEL_FAST, config.RETRY_TEMPERATURE, "oneshot", True)
                      for i in range(config.RETRY_SAMPLES)]
        elif config.ENABLE_REACT and config.ENABLE_REASONING_MODEL:
            rungs.append(("react-reason", "react", config.MODEL_REASON, 0.0, "react", False))
        tried = set(task.tried)
        for key, kind, model, temp, strategy, fb in rungs:
            # a deterministic rung phase 1 already ran (same model, no feedback) would only repeat itself
            if f"retry:{key}" in tried or (not fb and temp == 0.0 and f"{strategy}/{model}" in tried):
                continue
            return key, kind, model, temp, strategy, fb
        return None

    def _rung_cost(self, task, rung):
        _, kind, _, _, strategy, _ = rung
        if kind == "react":
            return self.est.predict(task, "react")["cost"]
        prompt = task.prompt
        return self.est.predict(task, strategy, prompt.text, prompt.target_text)["cost"]

    async def _retry(self, task, rung, sem):
        key, kind, model, temp, strategy, fb = rung
        async with sem:
            if self.time_left() < 60 or self.ledger.remaining() <= 0 or self.llm.dead:
                return False
            spent = self.ledger.spent
            try:
                if kind == "react":
                    res = await self._react(task, model, key=f"retry:{key}")
                else:
                    extra = None
                    if fb and getattr(task, "last_reply", ""):
                        extra = [{"role": "assistant", "content": task.last_reply},
                                 {"role": "user", "content": f"That did not work: {task.reason[:600]}\nFix it. "
                                                             f"Same output format."}]
                    res = await self._step(task, strategy, task.prompt, model, extra, temperature=temp,
                                           key=f"retry:{key}")
                if res is not None and res.status in (SOLVED, INFEASIBLE):
                    self._apply(task, res, f"retry:{key}")
                else:
                    task.status, task.strategy = "failed", f"retry:{key}"
            except Exception as e:  # noqa: BLE001
                task.status, task.reason = "failed", f"retry error: {e!r}"[:300]
                log(f"{task.id}: retry error {e!r}")
            if f"retry:{key}" not in task.tried:  # skipped before the call (reservation): never offer it again
                task.tried.append(f"retry:{key}")
            task.cost = self.ledger.task_cost(task.id)
            self.results.save()
            log(f"{task.id}: retry {key} -> {task.status} answer={task.answer!r} cost={task.cost:.0f} "
                f"spent={self.ledger.spent:.0f}")
            return self.ledger.spent > spent or task.status == "solved"

    async def phase2(self):
        if not config.ENABLE_RETRY or self.llm is None:
            return
        sem = asyncio.Semaphore(config.MAX_PARALLEL)
        # tasks phase 1 skipped while other workers held the budget get their normal ladder first
        skipped = [t for t in self.tasks if t.status == "pending" and not t.laddered]
        if skipped and self.ledger.remaining() > 0 and not self.llm.dead:
            skipped.sort(key=lambda t: -t.reward / max(1.0, t.predicted.get("expected_total", 1.0)))
            log(f"phase2: {len(skipped)} skipped tasks get their ladder, budget left {self.ledger.remaining():.0f}")
            await asyncio.gather(*(self._worker(t, sem) for t in skipped), return_exceptions=True)
        for rnd in range(1, config.RETRY_ROUNDS + 1):
            if self.time_left() < 60 or self.llm.dead or self.ledger.remaining() <= 0:
                break
            todo = []
            for t in self.tasks:
                if t.status != "failed":  # solved / infeasible are final, pending ones never got a ladder
                    continue
                rung = self._next_rung(t)
                if rung is None:
                    continue
                cost = self._rung_cost(t, rung)
                if cost <= self.ledger.remaining():
                    todo.append((t.reward / max(1.0, cost), t, rung))
            if not todo:
                break
            todo.sort(key=lambda x: -x[0])
            log(f"phase2 round {rnd}: {len(todo)} failed tasks, budget left {self.ledger.remaining():.0f}")
            moved = await asyncio.gather(*(self._retry(t, rung, sem) for _, t, rung in todo), return_exceptions=True)
            if not any(m is True for m in moved):
                break

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

    async def _llm_phases(self):
        await self.phase1()
        await self.phase2()

    def run(self):
        self.phase0()
        try:
            asyncio.run(asyncio.wait_for(self._llm_phases(), timeout=max(30, self.time_left())))
        except asyncio.TimeoutError:
            log("time limit reached")
        except Exception as e:  # noqa: BLE001 - answers already written stay valid
            log(f"LLM phase aborted: {e!r}")
        self.results.save()
        pts = sum(t.reward for t in self.tasks if t.status == "solved")
        log(f"done: solved {sum(t.status == 'solved' for t in self.tasks)}/{len(self.tasks)}, "
            f"points {pts:g}, tokens {self.ledger.spent:.0f}/{self.ledger.budget:.0f}, "
            f"{time.time() - self.t0:.0f}s")
