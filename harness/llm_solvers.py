"""Cheap LLM strategies (one model call + local verification) per task type.

Each strategy = build a compact prompt -> one call -> verify locally:
  math / data_analysis : model writes Python, we execute it in the task dir
  search               : compact KB dump in the prompt, answer must occur in KB
  coding               : model returns the whole fixed file, tests must pass
A strategy can be re-run on the reasoning model (`model=` argument) as the
first escalation step. "INFEASIBLE" replies are respected (no answer file).
"""
import os
import re

from . import sandbox
from .solvers import coding_solver, search_solver
from .solvers.base import SolveResult

INFEASIBLE_RX = re.compile(r"^\W*INFEASIBLE\b", re.I)
_TAIL = re.compile(r"\s*(Compute it with|Use bash|Use the file tools|Use the bash|Use python|Work only inside|"
                   r"When you are done)\b.*$", re.I | re.S)

SYS_CODE = ("You solve data tasks by writing Python 3. Reply with exactly one ```python code block that prints "
            "ONLY the final answer (a bare number or string, no words). If the task cannot be answered exactly "
            "from the given information (missing quantity, unknown column/value, impossible range, division by "
            "zero), reply with the single word INFEASIBLE instead.")


def problem_text(task):
    return _TAIL.sub("", task.instruction).strip()


class Prompt:
    def __init__(self, messages, max_tokens, target_text=""):
        self.messages = messages
        self.max_tokens = max_tokens
        self.target_text = target_text

    @property
    def text(self):
        return "\n".join(m["content"] for m in self.messages)


# ------------------------------------------------------------------ math
def build_math(task):
    return Prompt([{"role": "system", "content": SYS_CODE},
                   {"role": "user", "content": problem_text(task) + "\nUse integer arithmetic; 'rounded down' "
                                                                    "means floor division (//)."}], 400)


def finish_code(task, text, cwd):
    if INFEASIBLE_RX.match(text or ""):
        return SolveResult.infeasible("model: INFEASIBLE")
    code = sandbox.extract_code(text)
    if not code:
        return SolveResult.unknown("no code in reply")
    ok, last, out = sandbox.run_python(code, cwd)
    if not ok:
        return SolveResult.unknown(f"code failed: {out[-300:]}")
    ans = last.strip().strip("'\"")
    if INFEASIBLE_RX.match(ans):
        return SolveResult.infeasible("code printed INFEASIBLE")
    if re.fullmatch(r"-?\d+\.0+", ans):
        ans = ans.split(".")[0]
    return SolveResult.solved(ans, "llm code")


# ------------------------------------------------------------------ data analysis
def _csv_summary(path, max_rows=5):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "(empty file)"
    header, body = rows[0], rows[1:]
    lines = [",".join(header)] + [",".join(r) for r in body[:max_rows]]
    info = [f"{len(body)} data rows. First rows:"] + lines
    for i, col in enumerate(header):
        vals = [r[i] for r in body if i < len(r)]
        try:
            nums = [float(v) for v in vals]
            info.append(f"{col}: numeric {min(nums):g}..{max(nums):g}")
        except ValueError:
            uniq = sorted(set(vals))
            info.append(f"{col}: {len(uniq)} distinct: {', '.join(uniq[:25])}{' ...' if len(uniq) > 25 else ''}")
    return "\n".join(info)


def build_data(task):
    ws = task.cwd
    files = [f for f in sorted(os.listdir(ws)) if f.endswith((".csv", ".tsv"))]
    parts = []
    for f in files[:3]:
        try:
            parts.append(f"File {f}: " + _csv_summary(os.path.join(ws, f)))
        except Exception as e:  # noqa: BLE001
            parts.append(f"File {f}: unreadable ({e})")
    if not files:
        parts.append("Files in directory: " + ", ".join(sorted(os.listdir(ws))[:30]))
    user = "\n".join(parts) + "\n\nTask: " + problem_text(task) + \
        "\nThe script runs in the directory containing the files; read them with the csv module. If a value " \
        "named in the task (product, region, ...) never occurs in the file, or a needed column does not exist, " \
        "the task is INFEASIBLE."
    return Prompt([{"role": "system", "content": SYS_CODE}, {"role": "user", "content": user}], 500)


# ------------------------------------------------------------------ search
def _kb_lines(kb_dir):
    ents = search_solver.load_kb(kb_dir)
    lines = []
    for e in ents.values():
        attrs = "; ".join(f"{k.title()}: {v}" for k, v in e.items() if not k.startswith("_"))
        lines.append(f"{e['_title']} | {attrs}")
    return lines, ents


def build_search(task):
    kb_dir = os.path.join(task.cwd, "kb")
    m = re.search(r"The directory (\S+?)/? contains", task.instruction)
    if m:
        kb_dir = os.path.join(task.cwd, m[1])
    lines, _ = _kb_lines(kb_dir) if os.path.isdir(kb_dir) else ([], {})
    kb = "\n".join(lines)
    if len(kb) > 40000:  # keep the prompt small: entries sharing a capitalised word with the question
        words = set(re.findall(r"[A-Z][a-z]+", problem_text(task)))
        keep = [ln for ln in lines if words & set(re.findall(r"[A-Z][a-z]+", ln))]
        kb = "\n".join(keep[:400])
    sys_p = ("Answer questions using ONLY the knowledge base below (one entry per line: Name | Key: Value; ...). "
             "Follow references between entries step by step. Reply in at most 3 short lines; the last line must "
             "be 'ANSWER: <exact value as written in the knowledge base>'. If the knowledge base lacks an entry or "
             "field needed for the answer, the last line must be 'ANSWER: INFEASIBLE'.")
    user = f"Knowledge base:\n{kb}\n\nQuestion: {problem_text(task)}"
    return Prompt([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], 160)


def finish_search(task, text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    ans = (m[-1] if m else (text or "").strip().splitlines()[-1] if text and text.strip() else "").strip()
    ans = ans.strip().strip("*`'\".").strip()
    if not ans:
        return SolveResult.unknown("empty reply")
    if INFEASIBLE_RX.match(ans):
        return SolveResult.infeasible("model: INFEASIBLE")
    # the answer must be a value that literally occurs in the KB
    kb_dir = os.path.join(task.cwd, "kb")
    blob = ""
    for root, _, files in os.walk(kb_dir if os.path.isdir(kb_dir) else task.cwd):
        for fn in files:
            try:
                blob += open(os.path.join(root, fn), encoding="utf-8").read() + "\n"
            except (OSError, UnicodeDecodeError):
                pass
    if ans not in blob:
        return SolveResult.unknown(f"answer {ans!r} not found in kb")
    return SolveResult.solved(ans, "llm kb lookup")


# ------------------------------------------------------------------ coding
def build_coding(task, partial_src=None):
    ws = task.cwd
    test_file, target = coding_solver.parse_names(task.data, ws)
    if not test_file or not target:
        return None
    src = open(os.path.join(ws, target), encoding="utf-8").read()
    tests = open(os.path.join(ws, test_file), encoding="utf-8").read()
    _, out = coding_solver.run_tests(ws, test_file)
    out = "\n".join(ln for ln in out.splitlines() if ln.strip())[-1500:]
    sys_p = ("You fix bugs in Python code so that the given unit tests pass, without changing the tests. Keep "
             "function names/signatures and fix the logic so it is correct in general, not only for the tested "
             "inputs. Reply with the complete corrected file in one ```python code block and nothing else. If the "
             "tests contradict each other so that no implementation can pass them all, reply INFEASIBLE.")
    user = f"File {target}:\n```python\n{src}\n```\n\nTests {test_file}:\n```python\n{tests}\n```\n\n" \
           f"Test output:\n{out}"
    return Prompt([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                  min(4000, int(len(src) / 3.0) + 300), target_text=src)


def finish_coding(task, text):
    if INFEASIBLE_RX.match(text or ""):
        return SolveResult.infeasible("model: INFEASIBLE")
    code = sandbox.extract_code(text)
    if not code:
        return SolveResult.unknown("no code block")
    ws = task.cwd
    test_file, target = coding_solver.parse_names(task.data, ws)
    spath, tpath = os.path.join(ws, target), os.path.join(ws, test_file)
    original = open(spath, encoding="utf-8").read()
    tests_before = open(tpath, encoding="utf-8").read()
    with open(spath, "w", encoding="utf-8") as f:
        f.write(code.rstrip() + "\n")
    ok, out = coding_solver.run_tests(ws, test_file)
    tests_after = open(tpath, encoding="utf-8").read()
    if tests_after != tests_before:
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(tests_before)
        ok = False
    if not ok:
        with open(spath, "w", encoding="utf-8") as f:
            f.write(original)
        return SolveResult.unknown(f"tests still fail: {out[-400:]}")
    return SolveResult.solved("DONE", "llm patch")


BUILDERS = {"math": build_math, "data_analysis": build_data, "search": build_search, "coding": build_coding}


def build(task):
    fn = BUILDERS.get(task.type)
    if fn is None:
        return None
    try:
        return fn(task)
    except Exception:  # noqa: BLE001 - broken workspace -> let the ReAct agent look at it
        return None


def finish(task, text):
    if task.type == "search":
        return finish_search(task, text)
    if task.type == "coding":
        return finish_coding(task, text)
    return finish_code(task, text, task.cwd)
