"""Cheap LLM strategies (one model call + local verification) per task type.

Each strategy = build a compact prompt -> one call -> verify locally:
  math          : model writes ONE Python expression, we evaluate it
  data_analysis : model writes a short script, we execute it in the task dir
  search        : only the KB sub-graph reachable from the question's entity,
                  answer must occur in the KB
  coding        : only the functions used by failing tests + those tests; the
                  model returns the fixed functions, we splice them in and the
                  task's own tests must pass
A strategy can be re-run on the reasoning model (`model=` argument) as the
first escalation step. "INFEASIBLE" replies are respected (no answer file).
"""
import ast
import json
import os
import re

from . import sandbox
from .fsutil import atomic_write
from .solvers import coding_solver, math_solver, search_solver
from .solvers.base import SolveResult

INFEASIBLE_RX = re.compile(r"^\W*INFEASIBLE\b", re.I)
_TAIL = re.compile(r"\s*(Compute it with|Use bash|Use the file tools|Use the bash|Use python|Work only inside|"
                   r"When you are done)\b.*$", re.I | re.S)

SYS_CODE = ("Write a minimal Python 3 script (no comments) in one ```python block that prints ONLY the final "
            "answer. If the task cannot be answered exactly from the data (unknown column or value, impossible "
            "range), reply INFEASIBLE.")
SYS_STEPS = ("Translate the problem into Python, one line per sentence, in order: `s = <start>` for the first "
             "sentence, then exactly one `s = ...` line for each later statement, then `print(s)`. Integers only; "
             "'rounded down' = //, applied to the amount the sentence names (selling 25% of s, rounded down: "
             "`s = s - s * 25 // 100`; keeping one share of k: `s = s // k`). Output only the code. If a needed "
             "quantity is not given or the problem is impossible, reply INFEASIBLE.")


_FILE_DECL = re.compile(r"The file \S+ has columns [\w,]+(?: \(\d+ rows?\))?\.\s*", re.I)
_KB_DECL = re.compile(r"The directory \S+ contains a small knowledge base of markdown files\.\s*", re.I)


def problem_text(task):
    """Instruction without tool/IO boilerplate (and without context the prompt
    already shows in compact form)."""
    t = _TAIL.sub("", task.instruction).strip()
    return _KB_DECL.sub("", _FILE_DECL.sub("", t)).strip()


class Prompt:
    def __init__(self, messages, max_tokens, target_text=""):
        self.messages = messages
        self.max_tokens = max_tokens
        self.target_text = target_text

    @property
    def text(self):
        return "\n".join(m["content"] for m in self.messages)


# ------------------------------------------------------------------ math
def _math_statements(task):
    """Statements in the word problem (every sentence except the closing question)."""
    return max(1, len(math_solver._sentences(problem_text(task))) - 1)


# Expected size of one generated `s = ...` line; the estimator prices output tokens from it.
_STEP_LINE_SIZE = " " * 60


def build_math(task):
    n = _math_statements(task)
    return Prompt([{"role": "system", "content": SYS_STEPS},
                   {"role": "user", "content": problem_text(task)}],
                  min(600, 40 * n + 60), target_text=_STEP_LINE_SIZE * n)


def finish_math(task, text):
    """A fast model sometimes stops mid-chain: every statement must have produced a line."""
    if not INFEASIBLE_RX.match(text or ""):
        code = sandbox.extract_code(text) or text or ""
        lines = len(re.findall(r"^\s*s\s*[-+*/%]*=(?!=)", code, re.M))
        n = _math_statements(task)
        if lines and lines < n:
            return SolveResult.unknown(f"only {lines} steps for {n} statements: write one line per sentence")
    res = finish_code(task, text, task.cwd)
    if res.status == "solved" and not re.fullmatch(r"-?\d+(?:\.\d+)?", str(res.answer).strip()):
        return SolveResult.unknown(f"the answer must be a number, got {str(res.answer)[:40]!r}")
    return res


def _as_program(text):
    """Expression / snippet / code block -> a program that prints the answer."""
    code = sandbox.extract_code(text) or (text or "").strip().strip("`").strip()
    if not code:
        return None
    if "print(" in code:
        return code
    lines = [ln for ln in code.splitlines() if ln.strip()]
    lines[-1] = f"print({lines[-1].strip()})"
    return "\n".join(lines)


def finish_code(task, text, cwd, preamble=""):
    if INFEASIBLE_RX.match(text or ""):
        return SolveResult.infeasible("model: INFEASIBLE")
    code = _as_program(text)
    if not code:
        return SolveResult.unknown("no code in reply")
    ok, last, out = sandbox.run_python(preamble + code, cwd)
    if not ok:
        return SolveResult.unknown(f"code failed: {out[-300:]}")
    ans = last.strip().strip("'\"")
    if INFEASIBLE_RX.match(ans):
        return SolveResult.infeasible("code printed INFEASIBLE")
    if re.fullmatch(r"-?\d+\.0+", ans):
        ans = ans.split(".")[0]
    return SolveResult.solved(ans, "llm code")


# ------------------------------------------------------------------ data analysis
def _csv_summary(path, max_rows=3):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "(empty file)"
    header, body = rows[0], rows[1:]
    lines = [",".join(header)] + [",".join(r) for r in body[:max_rows]]
    info = [f"{len(body)} rows; first rows:"] + lines
    for i, col in enumerate(header):
        vals = [r[i] for r in body if i < len(r)]
        try:
            nums = [float(v) for v in vals]
            info.append(f"{col}: {min(nums):g}..{max(nums):g}")
        except ValueError:
            uniq = sorted(set(vals))
            info.append(f"{col}: {', '.join(uniq[:25])}{' ...' if len(uniq) > 25 else ''}")
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
    user = "\n".join(parts) + "\n\nTask (the files are in the current directory): " + problem_text(task)
    return Prompt([{"role": "system", "content": SYS_CODE}, {"role": "user", "content": user}], 300)


# ------------------------------------------------------------------ search
def _kb_dir(task):
    m = re.search(r"The directory (\S+?)/? contains", task.instruction)
    return os.path.join(task.cwd, m[1] if m else "kb")


def _subgraph(ents, question, depth=3):
    """Entities reachable from the names in the question (follows values that
    are themselves entity names). Empty anchor -> the whole KB."""
    ql = question.lower()
    frontier = [k for k in ents if re.search(r"\b" + re.escape(k) + r"\b", ql)]
    if not frontier:
        return list(ents)
    seen = list(dict.fromkeys(frontier))
    for _ in range(depth):
        nxt = []
        for k in frontier:
            for a, v in ents[k].items():
                if not a.startswith("_") and v.lower() in ents and v.lower() not in seen:
                    seen.append(v.lower())
                    nxt.append(v.lower())
        frontier = nxt
    return seen


def _kb_records(ents):
    """KB entries as plain records; numeric fields become ints so comparisons just work."""
    out = []
    for e in ents.values():
        rec = {"name": e["_title"], "kind": e["_kind"]}
        for a, v in e.items():
            if not a.startswith("_"):
                rec[a] = int(v) if re.fullmatch(r"-?\d+", v) else v
        out.append(rec)
    return out


# Expected size of a short query script; the estimator prices output tokens from it.
_QUERY_SCRIPT_SIZE = " " * 280


def _build_search_query(task, ents, question):
    """Counts, reverse lookups and comparisons need the whole KB. The model gets only the
    schema and writes a query; the harness runs it over all records."""
    schema, records = {}, _kb_records(ents)
    for rec in records:
        schema.setdefault(rec["kind"], rec)
    lines = [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in schema.items()]
    sys_p = ("`E` is a predefined Python list of dicts holding EVERY knowledge-base entry. Only one record per kind is "
             "shown, to give the format: people and companies named in the question are in E even if not shown. "
             "Write a minimal Python 3 script (no comments, do not define E) in one ```python block that looks them "
             "up in E and prints ONLY the final answer. Reply INFEASIBLE only if a field the question needs does not "
             "exist in the shown format.")
    user = f"E has {len(records)} records. Format per kind:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"
    return Prompt([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], 300,
                  target_text=_QUERY_SCRIPT_SIZE)


def build_search(task):
    kb_dir = _kb_dir(task)
    ents = search_solver.load_kb(kb_dir) if os.path.isdir(kb_dir) else {}
    question = problem_text(task)
    if ents and search_solver.is_non_chain(question):
        return _build_search_query(task, ents, question)
    lines = []
    for k in _subgraph(ents, question):
        e = ents[k]
        attrs = "; ".join(f"{a.title()}: {v}" for a, v in e.items() if not a.startswith("_"))
        lines.append(f"{e['_title']} | {attrs}")
    kb = "\n".join(lines[:400])
    sys_p = ("Answer from this knowledge base only (Name | Key: Value). Follow references step by step. At most 2 "
             "short lines; the last is 'ANSWER: <value exactly as written>' or 'ANSWER: INFEASIBLE' if an entry or "
             "field needed is missing.")
    user = f"{kb}\n\nQuestion: {question}"
    return Prompt([{"role": "system", "content": sys_p}, {"role": "user", "content": user}], 80)


def finish_search(task, text):
    kb_dir = _kb_dir(task)
    if os.path.isdir(kb_dir) and search_solver.is_non_chain(problem_text(task)):
        data = json.dumps(_kb_records(search_solver.load_kb(kb_dir)), ensure_ascii=False)
        return finish_code(task, text, task.cwd, preamble=f"import json\nE = json.loads({data!r})\n")
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    ans = (m[-1] if m else (text or "").strip().splitlines()[-1] if text and text.strip() else "").strip()
    ans = ans.strip().strip("*`'\".").strip()
    if not ans:
        return SolveResult.unknown("empty reply")
    if INFEASIBLE_RX.match(ans):
        return SolveResult.infeasible("model: INFEASIBLE")
    # the answer must be a value that literally occurs in the KB
    kb_dir = _kb_dir(task)
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
def _defs(tree):
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _implicated(src, tests, out):
    """(function sources, failing test sources) for the failing tests, or None."""
    failing = re.findall(r"^(?:FAIL|ERROR): (\w+)", out, re.M)
    try:
        stree, ttree = ast.parse(src), ast.parse(tests)
    except SyntaxError:
        return None
    if any(isinstance(n, ast.ClassDef) for n in stree.body):
        return None  # methods / classes: give the model the whole file
    funcs = _defs(stree)
    if not failing or not funcs:
        return None
    tsrc, used = [], []
    for node in ast.walk(ttree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in failing:
            tsrc.append(ast.get_source_segment(tests, node))
            for c in ast.walk(node):
                name = c.attr if isinstance(c, ast.Attribute) else c.id if isinstance(c, ast.Name) else None
                if name in funcs and name not in used:
                    used.append(name)
    if not used or len(tsrc) != len(set(failing)):
        return None
    return [ast.get_source_segment(src, funcs[n]) for n in used], tsrc


def _rival_hint(task):
    """Candidate repairs the zero-token search found ambiguous (they all pass the
    visible tests but behave differently) - let the model pick by intent."""
    rivals = [r for r in (getattr(task, "det_info", {}) or {}).get("rivals") or [] if r][:4]
    if len(rivals) < 2:
        return ""
    return ("\nThese one-token repairs all pass the visible test but behave differently on other inputs; at most "
            "one is right:\n```python\n" + "\n\n".join(rivals) + "\n```")


def _failing(out):
    return set(re.findall(r"^(?:FAIL|ERROR): (\w+)", out, re.M))


def build_coding(task):
    ws = task.cwd
    test_file, target = coding_solver.parse_names(task.data, ws)
    if not test_file or not target:
        return None
    src = open(os.path.join(ws, target), encoding="utf-8").read()
    tests = open(os.path.join(ws, test_file), encoding="utf-8").read()
    _, out = coding_solver.run_tests(ws, test_file)
    errors = "\n".join(ln.strip() for ln in out.splitlines()
                       if re.match(r"\s*(FAIL|ERROR|\w*Error|AssertionError)\b", ln))[-800:]
    imp = _implicated(src, tests, out)
    if imp is not None:
        fsrc, tsrc = imp
        imports = "\n".join(ln for ln in src.splitlines() if re.match(r"(import|from) \w", ln))
        body = "\n\n".join(fsrc)
        sys_p = ("Fix the bugs so the failing tests pass; keep signatures. Hidden tests use other inputs, so each "
                 "function must do exactly what its name says, not merely pass these examples: check every operator, "
                 "constant and index, a function can contain more than one bug. Reply with only the corrected "
                 "function(s) in one ```python block. If the tests contradict each other, reply INFEASIBLE.")
        user = (f"From {target}:\n```python\n{(imports + chr(10) * 2) if imports else ''}{body}\n```\n"
                f"Failing tests:\n```python\n" + "\n".join(tsrc) + f"\n```\n{errors}" + _rival_hint(task))
        return Prompt([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                      min(2000, int(len(body) / 2.5) + 120), target_text=body)
    sys_p = ("Fix the bugs so the tests pass without changing the tests; keep signatures. Hidden tests use other "
             "inputs, so each function must do exactly what its name says: check every operator, constant and "
             "index, a function can contain more than one bug. Reply with the complete corrected file in one "
             "```python block. If the tests contradict each other, reply INFEASIBLE.")
    user = f"{target}:\n```python\n{src}\n```\n{test_file}:\n```python\n{tests}\n```\n{errors}" + _rival_hint(task)
    return Prompt([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                  min(4000, int(len(src) / 2.5) + 200), target_text=src)


def _splice(original, code):
    """Replace the functions defined in `code` inside `original` (append new ones,
    add missing imports). A full-file reply therefore works too."""
    otree, ctree = ast.parse(original), ast.parse(code)
    odefs = _defs(otree)
    new = original
    for name, node in _defs(ctree).items():
        fsrc = ast.get_source_segment(code, node)
        if node.decorator_list:
            fsrc = "\n".join("@" + ast.unparse(d) for d in node.decorator_list) + "\n" + fsrc
        if name in odefs:
            new = coding_solver._replace_function(new, _defs(ast.parse(new))[name], fsrc)
        else:
            new = new.rstrip() + "\n\n\n" + fsrc + "\n"
    for n in ctree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            line = ast.unparse(n)
            if line not in new:
                new = line + "\n" + new
    if not _defs(ctree):  # no function in the reply -> treat as a whole file
        new = code
    return new


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
    _, out0 = coding_solver.run_tests(ws, test_file)
    failing_before = _failing(out0)
    try:
        patched = _splice(original, code)
    except (SyntaxError, IndexError, ValueError) as e:
        return SolveResult.unknown(f"cannot apply patch: {e}")
    atomic_write(spath, patched.rstrip() + "\n")
    ok, out = coding_solver.run_tests(ws, test_file)
    tests_after = open(tpath, encoding="utf-8").read()
    if tests_after != tests_before:
        atomic_write(tpath, tests_before)
        ok = False
    if not ok:
        atomic_write(spath, original)
        errs = "\n".join(ln for ln in out.splitlines() if re.match(r"\s*(FAIL|ERROR|\w*Error)", ln))
        failing_after = _failing(out)
        # every originally failing test now passes but a passing one broke: the
        # tests may be pulling in opposite directions (see the ladder)
        tradeoff = bool(failing_before) and bool(failing_after) and not (failing_before & failing_after)
        return SolveResult.unknown(f"tests still fail: {errs[-400:]}", tradeoff=tradeoff)
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
    if task.type == "math":
        return finish_math(task, text)
    return finish_code(task, text, task.cwd)
