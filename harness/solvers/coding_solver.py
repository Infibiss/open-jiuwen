"""Zero-token repair of injected bugs via AST mutation search.

The bugs in these tasks are small mutations of correct code (`-=` instead of
`+=`, `<` instead of `>`, `sorted` instead of `reversed`, `% 2 == 1`, ...).
We enumerate inverse mutations of every function, evaluate them in a sandboxed
subprocess against the *unchanged* test file and greedily keep the mutation
that removes the most failing tests without breaking passing ones. The final
file is verified with the task's own command (`python3 test_mylib.py`).
"""
import ast
import copy
import json
import os
import re
import subprocess
import sys

from .base import SolveResult

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_runner.py")

BIN_ALT = {
    ast.Add: [ast.Sub, ast.Mult], ast.Sub: [ast.Add], ast.Mult: [ast.Add, ast.FloorDiv, ast.Div, ast.Pow],
    ast.Div: [ast.FloorDiv, ast.Mult], ast.FloorDiv: [ast.Div, ast.Mod, ast.Mult], ast.Mod: [ast.FloorDiv],
    ast.Pow: [ast.Mult],
}
CMP_ALT = {
    ast.Lt: [ast.Gt, ast.LtE, ast.GtE], ast.Gt: [ast.Lt, ast.GtE, ast.LtE],
    ast.LtE: [ast.GtE, ast.Lt, ast.Gt], ast.GtE: [ast.LtE, ast.Gt, ast.Lt],
    ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.In: [ast.NotIn], ast.NotIn: [ast.In],
    ast.Is: [ast.IsNot], ast.IsNot: [ast.Is],
}
NAME_ALT = {
    "min": ["max"], "max": ["min"], "sorted": ["reversed"], "reversed": ["sorted"], "any": ["all"], "all": ["any"],
}
ATTR_ALT = {
    "upper": ["lower"], "lower": ["upper"], "startswith": ["endswith"], "endswith": ["startswith"],
    "lstrip": ["rstrip", "strip"], "rstrip": ["lstrip", "strip"], "strip": ["lstrip", "rstrip"],
    "find": ["rfind"], "rfind": ["find"], "index": ["rindex"], "title": ["capitalize"], "capitalize": ["title"],
    "isupper": ["islower"], "islower": ["isupper"], "append": ["insert"], "remove": ["discard"],
}


def _mutants_of_node(node):
    """Yield callables that mutate `node` in place (applied on a deep copy)."""
    if isinstance(node, (ast.BinOp, ast.AugAssign)):
        for alt in BIN_ALT.get(type(node.op), []):
            yield lambda n, alt=alt: setattr(n, "op", alt())
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod)):
        def swap(n):
            n.left, n.right = n.right, n.left
        yield swap
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            for alt in CMP_ALT.get(type(op), []):
                def f(n, i=i, alt=alt):
                    n.ops[i] = alt()
                yield f
    if isinstance(node, ast.BoolOp):
        yield lambda n: setattr(n, "op", ast.Or() if isinstance(n.op, ast.And) else ast.And())
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            yield lambda n: setattr(n, "value", not n.value)
        elif isinstance(v, int):
            for alt in dict.fromkeys([v + 1, v - 1, 0, 1, -v]):
                if alt != v:
                    yield lambda n, alt=alt: setattr(n, "value", alt)
        elif isinstance(v, float):
            for alt in (0.0, 1.0, v + 1, v - 1):
                if alt != v:
                    yield lambda n, alt=alt: setattr(n, "value", alt)
        elif isinstance(v, str) and v in ("", " ", ",", ", "):
            for alt in ("", " ", ",", ", "):
                if alt != v:
                    yield lambda n, alt=alt: setattr(n, "value", alt)
    if isinstance(node, ast.Name) and node.id in NAME_ALT:
        for alt in NAME_ALT[node.id]:
            yield lambda n, alt=alt: setattr(n, "id", alt)
    if isinstance(node, ast.Name) and node.id in ("True", "False"):
        yield lambda n: setattr(n, "id", "False" if n.id == "True" else "True")
    if isinstance(node, ast.Attribute) and node.attr in ATTR_ALT:
        for alt in ATTR_ALT[node.attr]:
            yield lambda n, alt=alt: setattr(n, "attr", alt)
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "sorted":
            kws = [k for k in node.keywords if k.arg == "reverse"]
            if not kws:
                yield lambda n: n.keywords.append(ast.keyword(arg="reverse", value=ast.Constant(True)))
        for i, kw in enumerate(node.keywords):
            if kw.arg == "reverse" and isinstance(kw.value, ast.Constant):
                yield lambda n, i=i: n.keywords.pop(i)
        if isinstance(fn, ast.Attribute) and node.args and fn.attr in ("split", "strip", "lstrip", "rstrip", "pop"):
            yield lambda n: n.args.clear()  # s.split(' ') -> s.split()
        if isinstance(fn, ast.Name) and fn.id == "range" and node.args:
            def plus1(n):
                n.args[-1 if len(n.args) < 3 else 1] = ast.BinOp(
                    n.args[-1 if len(n.args) < 3 else 1], ast.Add(), ast.Constant(1))
            yield plus1

            def minus1(n):
                n.args[-1 if len(n.args) < 3 else 1] = ast.BinOp(
                    n.args[-1 if len(n.args) < 3 else 1], ast.Sub(), ast.Constant(1))
            yield minus1
            if len(node.args) == 1:
                yield lambda n: n.args.insert(0, ast.Constant(1))
    if isinstance(node, ast.BinOp) and isinstance(node.right, ast.Constant) and node.right.value in (1, -1) \
            and isinstance(node.op, (ast.Add, ast.Sub)):
        yield "UNWRAP_LEFT"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
        yield "UNWRAP_OPERAND"
    if isinstance(node, ast.Slice):
        for attr in ("lower", "upper"):
            b = getattr(node, attr)
            if b is not None:
                yield lambda n, attr=attr: setattr(n, attr, ast.BinOp(getattr(n, attr), ast.Add(), ast.Constant(1)))
                yield lambda n, attr=attr: setattr(n, attr, ast.BinOp(getattr(n, attr), ast.Sub(), ast.Constant(1)))
                yield lambda n, attr=attr: setattr(n, attr, None)
        if node.step is not None:
            yield lambda n: setattr(n, "step", None)
        else:
            yield lambda n: setattr(n, "step", ast.UnaryOp(ast.USub(), ast.Constant(1)))
    if isinstance(node, ast.Return) and node.value is not None:
        yield lambda n: setattr(n, "value", ast.UnaryOp(ast.Not(), n.value))


class _Unwrap(ast.NodeTransformer):
    """Replace the k-th visited node by one of its children."""

    def __init__(self, target, mode):
        self.target, self.mode = target, mode

    def generic_visit(self, node):
        for field, old in ast.iter_fields(node):
            if isinstance(old, list):
                for i, v in enumerate(old):
                    if v is self.target:
                        old[i] = self._repl(v)
                    elif isinstance(v, ast.AST):
                        self.visit(v)
            elif isinstance(old, ast.AST):
                if old is self.target:
                    setattr(node, field, self._repl(old))
                else:
                    self.visit(old)
        return node

    def _repl(self, n):
        return n.left if self.mode == "UNWRAP_LEFT" else n.operand


def function_mutants(func_node):
    """All single-site mutants of a FunctionDef, as source strings (ordered)."""
    sites = list(ast.walk(func_node))
    out = []
    for idx, node in enumerate(sites):
        for mut in _mutants_of_node(node):
            fcopy = copy.deepcopy(func_node)
            target = list(ast.walk(fcopy))[idx]
            if isinstance(mut, str):
                _Unwrap(target, mut).visit(fcopy)
            else:
                mut(target)
            try:
                src = ast.unparse(ast.fix_missing_locations(fcopy))
                compile(src, "<mutant>", "exec")
            except Exception:  # noqa: BLE001 - invalid mutant
                continue
            out.append(src)
    return list(dict.fromkeys(out))


def _replace_function(src, func_node, new_func_src):
    lines = src.splitlines(keepends=True)
    start = (func_node.decorator_list[0].lineno if func_node.decorator_list else func_node.lineno) - 1
    end = func_node.end_lineno
    indent = re.match(r"\s*", lines[start]).group(0)
    body = "".join(indent + ln + "\n" if ln.strip() else "\n" for ln in new_func_src.splitlines())
    return "".join(lines[:start]) + body + "".join(lines[end:])


def _evaluate(workspace, test_file, module, candidates, timeout=120):
    if not candidates:
        return []
    req = json.dumps({"workspace": workspace, "test_file": test_file, "module": module, "candidates": candidates})
    p = subprocess.run([sys.executable, "-I", RUNNER], input=req, capture_output=True, text=True,
                       timeout=timeout, cwd=workspace)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return [{"ok": False, "failed": ["<runner>"]}] * len(candidates)


def _second_order(workspace, test_file, module, src, funcs, failing, cap=6000):
    """Two simultaneous mutations inside one implicated function (e.g. clamp:
    min(lo, max(x, hi)) -> max(lo, min(x, hi)))."""
    fail_text = " ".join(failing).lower()
    implicated = [f for f in funcs if f.name.lower() in fail_text] or funcs
    cands, meta = [], []
    for f in implicated:
        seen = set()
        for m1 in function_mutants(f):
            f1 = ast.parse(m1).body[0]
            for m2 in function_mutants(f1):
                if m2 in seen:
                    continue
                seen.add(m2)
                cands.append(_replace_function(src, f, m2))
                meta.append(f.name)
                if len(cands) >= cap:
                    break
    best = None
    for start in range(0, len(cands), 1500):
        results = _evaluate(workspace, test_file, module, cands[start:start + 1500])
        for j, r in enumerate(results):
            fset = set(r.get("failed", []))
            if "<import>" in fset or not fset <= failing:
                continue
            gain = len(failing) - len(fset)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, start + j, fset)
    return None if best is None else (*best, cands, meta)


def run_tests(workspace, test_file, timeout=120):
    """Run the task's own test command. Returns (passed, combined output)."""
    try:
        p = subprocess.run([sys.executable, test_file], cwd=workspace, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    out = (p.stdout + p.stderr)[-4000:]
    return p.returncode == 0, out


def contradictory_asserts(test_src):
    """Detect tests that can never pass together, e.g.
    assertEqual(m.f(x), 3) + assertNotEqual(m.f(x), 3)  or  assertTrue(e) + assertFalse(e)."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None
    eq, ne, tf = {}, set(), {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args):
            continue
        name, args = node.func.attr, node.args
        key = ast.unparse(args[0])
        if name in ("assertEqual", "assertEquals") and len(args) >= 2:
            try:
                val = ast.literal_eval(args[1])
            except Exception:  # noqa: BLE001 - non-literal expectation, ignore
                continue
            if key in eq and eq[key] != val:
                return f"{key} expected to equal both {eq[key]!r} and {val!r}"
            eq.setdefault(key, val)
        elif name in ("assertNotEqual", "assertNotEquals") and len(args) >= 2:
            try:
                ne.add((key, repr(ast.literal_eval(args[1]))))
            except Exception:  # noqa: BLE001
                continue
        elif name in ("assertTrue", "assertFalse"):
            want = name == "assertTrue"
            if tf.get(key, want) != want:
                return f"{key} asserted both true and false"
            tf[key] = want
    for key, val in eq.items():
        if (key, repr(val)) in ne:
            return f"{key} asserted both == and != {val!r}"
    return None


def parse_names(task, workspace):
    instr = task.get("instruction", "")
    tm = re.search(r"python3?\s+([\w./-]+\.py)", instr)
    test_file = tm[1] if tm else None
    fm = re.search(r"bugs? in ([\w./-]+\.py)", instr)
    target = fm[1] if fm else None
    files = sorted(os.listdir(workspace)) if workspace and os.path.isdir(workspace) else []
    if not test_file:
        tests = [f for f in files if f.startswith("test") and f.endswith(".py")]
        test_file = tests[0] if len(tests) == 1 else None
    if not target and test_file:
        cand = test_file[len("test_"):] if test_file.startswith("test_") else None
        target = cand if cand in files else None
    return test_file, target


def solve(task, workspace, max_rounds=12):
    if not workspace or not os.path.isdir(workspace):
        return SolveResult.unknown("no workspace")
    test_file, target = parse_names(task, workspace)
    if not test_file or not target:
        return SolveResult.unknown("cannot determine test/target files")
    tpath, spath = os.path.join(workspace, test_file), os.path.join(workspace, target)
    if not os.path.isfile(tpath):
        return SolveResult.infeasible(f"test file {test_file} missing")
    if not os.path.isfile(spath):
        return SolveResult.unknown(f"target {target} missing")
    module = os.path.splitext(os.path.basename(target))[0]
    why = contradictory_asserts(open(tpath, encoding="utf-8").read())
    if why:
        return SolveResult.infeasible(f"contradictory tests: {why}")
    original = open(spath, encoding="utf-8").read()

    ok, out = run_tests(workspace, test_file)
    if ok:
        return SolveResult.solved("DONE", "tests already pass", edits=0)

    src = original
    base = _evaluate(workspace, test_file, module, [src])[0]
    failing = set(base["failed"])
    if "<import>" in failing or "<crash>" in failing or "<runner>" in failing:
        return SolveResult.unknown("module/test import error", output=out[-800:])
    if not failing:
        return SolveResult.unknown("runner sees no failures but command fails", output=out[-800:])

    history = []
    for _ in range(max_rounds):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return SolveResult.unknown("syntax error in target")
        funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            funcs += [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        # functions named in failing tests first
        fail_text = " ".join(failing).lower()
        funcs.sort(key=lambda f: 0 if f.name.lower() in fail_text else 1)
        cands, meta = [], []
        for f in funcs:
            for msrc in function_mutants(f):
                cands.append(_replace_function(src, f, msrc))
                meta.append(f.name)
        results = _evaluate(workspace, test_file, module, cands)
        best = None
        for i, r in enumerate(results):
            fset = set(r.get("failed", []))
            if "<import>" in fset or not fset <= failing:
                continue  # breaks something that used to work
            gain = len(failing) - len(fset)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, i, fset)
        if best is None:
            best = _second_order(workspace, test_file, module, src, funcs, failing)
            if best is None:
                break
            cands, meta = best[3], best[4]
        _, i, failing = best[:3]
        src = cands[i]
        history.append(meta[i])
        if not failing:
            break

    if failing:
        return SolveResult.unknown(f"mutation search left {len(failing)} failing tests",
                                   failing=sorted(failing), partial_src=src, edits=history)
    with open(spath, "w", encoding="utf-8") as f:
        f.write(src)
    ok, out = run_tests(workspace, test_file)
    if not ok:
        with open(spath, "w", encoding="utf-8") as f:
            f.write(original)
        return SolveResult.unknown("candidate passed runner but not the real test command", output=out[-800:])
    return SolveResult.solved("DONE", f"fixed by {len(history)} mutation(s) in {sorted(set(history))}",
                              edits=history)
