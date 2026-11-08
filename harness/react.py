"""Last-resort agentic solver: openJiuwen ReActAgent with a confined bash tool.

Used only when the zero-token solver and the one-shot strategy failed (or for
unknown task types). Every model call is charged to the Ledger through the
AFTER_MODEL_CALL callback; the run is cancelled as soon as the task exceeds
its token cap, and the number of iterations is bounded.
"""
import asyncio
import os
import re
import uuid

from . import config
from .oj import AgentCallbackEvent, AgentCard, ReActAgent, ReActAgentConfig, ModelRequestConfig, tool
from .llm import client_config
from .solvers import coding_solver
from .solvers.base import SolveResult

SYSTEM = (
    "You are a precise, economical command-line agent. Solve the task in the current directory using the bash "
    "tool (python3 is available). Keep tool outputs short (use head/grep). Never edit test files. Never access "
    "the network or paths outside the current directory. When finished, write ONLY the final answer to "
    "answer.txt in the current directory and reply 'FINAL: <answer>'. If the task is impossible to answer "
    "exactly from the available data, do not create answer.txt and reply 'FINAL: INFEASIBLE'."
)


def _bash_tool(workdir, timeout_s=40, max_output=6000):
    async def bash(command: str) -> str:
        """Run a bash command in the task directory.

        Args:
            command: The bash command line to execute.
        """
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1"}
        proc = await asyncio.create_subprocess_shell(
            command, cwd=workdir, executable="/bin/bash", env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            return f"[exit=timeout after {timeout_s}s]"
        text = out.decode("utf-8", "replace")
        if len(text) > max_output:
            text = text[:max_output // 2] + "\n...[truncated]...\n" + text[-max_output // 2:]
        return f"[exit={proc.returncode}]\n{text}"

    t = tool(bash, name="bash", description="Run a bash command in the task directory; returns exit code and "
                                            "combined stdout/stderr.")
    t.card.properties["resilience"] = {"timeout_s": timeout_s + 5}
    return t


async def run_react(task, ledger, model, token_cap, max_iters=None):
    max_iters = max_iters or config.REACT_MAX_ITERS
    workdir = task.cwd
    test_file = target = tests_before = None
    if task.type == "coding":
        test_file, target = coding_solver.parse_names(task.data, workdir)
        if test_file and os.path.isfile(os.path.join(workdir, test_file)):
            tests_before = open(os.path.join(workdir, test_file), encoding="utf-8").read()
    ans_path = os.path.join(workdir, "answer.txt")
    if os.path.isfile(ans_path):
        os.remove(ans_path)

    req = ModelRequestConfig(model=model, temperature=0.0, max_tokens=1200 if model.endswith("noreason") else 3000)
    if model.endswith("-noreason"):
        req = ModelRequestConfig(model=model, temperature=0.0, max_tokens=1200,
                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    cfg = ReActAgentConfig(model_name=model, model_provider="OpenAI", api_key=config.LLM_API_KEY or "EMPTY",
                           api_base=config.LLM_BASE_URL, model_client_config=client_config(),
                           model_config_obj=req, prompt_template=[{"role": "system", "content": SYSTEM}],
                           max_iterations=max_iters, on_max_iterations="final_answer", parallel_tool_calls=False)
    agent = ReActAgent(card=AgentCard(name=f"solver_{task.id}", description="task solver"))
    agent.configure(cfg)
    bt = _bash_tool(workdir)
    agent.ability_manager.add_ability(bt.card, bt)

    over_cap = asyncio.Event()
    stop_result = {"output": "FINAL: STOPPED (token cap)", "result_type": "answer"}

    def _exhausted():
        return ledger.task_cost(task.id) >= token_cap or ledger.remaining() <= 0

    async def before_call(ctx):
        # openJiuwen's graceful abort: skips the model call and ends the loop
        if _exhausted():
            over_cap.set()
            ctx.request_force_finish(stop_result)

    async def after_call(ctx):
        resp = getattr(ctx.inputs, "response", None)
        um = getattr(resp, "usage_metadata", None)
        pin = getattr(um, "input_tokens", 0) or 0
        pout = getattr(um, "output_tokens", 0) or 0
        prs = getattr(um, "reasoning_tokens", 0) or 0
        ledger.charge(task.id, model, pin, pout, prs, not (prs and prs > pout), tag="react")
        if _exhausted():
            over_cap.set()
            ctx.request_force_finish(stop_result)

    await agent.register_callback(AgentCallbackEvent.BEFORE_MODEL_CALL, before_call)
    await agent.register_callback(AgentCallbackEvent.AFTER_MODEL_CALL, after_call)
    query = task.instruction
    # The token cap is enforced inside the loop (force_finish in the model-call
    # callbacks), so the run ends by itself; the wall-clock timeout is a backstop.
    run = asyncio.create_task(agent.invoke({"query": query, "conversation_id": f"{task.id}-{uuid.uuid4().hex[:6]}"}))
    done, _ = await asyncio.wait({run}, timeout=config.LLM_TIMEOUT_S * (max_iters + 1))
    result = None
    if run in done and not run.cancelled() and run.exception() is None:
        result = run.result()
    if not run.done():
        run.cancel()
    # restore tests if the agent touched them
    if tests_before is not None:
        tp = os.path.join(workdir, test_file)
        if open(tp, encoding="utf-8").read() != tests_before:
            with open(tp, "w", encoding="utf-8") as f:
                f.write(tests_before)

    output = ""
    if isinstance(result, dict):
        output = str(result.get("output") or "")
    final = [f for f in re.findall(r"FINAL:\s*(.+)", output) if not f.startswith("STOPPED")]
    file_ans = open(ans_path, encoding="utf-8").read().strip() if os.path.isfile(ans_path) else None
    if os.path.isfile(ans_path):
        os.remove(ans_path)  # the harness writes the official answer files itself

    if task.type == "coding":
        if not test_file:
            return SolveResult.unknown("react: no test file")
        ok, out = coding_solver.run_tests(workdir, test_file)
        if ok:
            return SolveResult.solved("DONE", "react agent")
        if final and final[-1].strip().upper().startswith("INFEASIBLE"):
            return SolveResult.infeasible("react: INFEASIBLE")
        return SolveResult.unknown(f"react: tests fail {out[-200:]}")
    if file_ans:
        return SolveResult.solved(file_ans.splitlines()[0].strip(), "react agent")
    if final:
        f = final[-1].strip().strip("`'\" ")
        if f.upper().startswith("INFEASIBLE"):
            return SolveResult.infeasible("react: INFEASIBLE")
        if f:
            return SolveResult.solved(f, "react agent (final text)")
    return SolveResult.unknown("react: no answer" + (" (cap)" if over_cap.is_set() else ""))
