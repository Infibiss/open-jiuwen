"""Zero-token repair of injected bugs via AST mutation search.

The bugs in these tasks are small mutations of correct code (`-=` instead of
`+=`, `<` instead of `>`, `sorted` instead of `reversed`, `% 2 == 1`, ...).
We enumerate inverse mutations of every function, evaluate them in a sandboxed
subprocess against the *unchanged* test file and greedily keep the mutation
that removes the most failing tests without breaking passing ones. The final
file is verified with the task's own command (`python3 test_mylib.py`).

Scoring uses hidden tests, and one visible assertion per function often lets a
wrong repair pass. So a repair is accepted without the LLM only when
  * every applied mutation is a pure inverse flip (SAFE: +/-, </>, ==/!=,
    and/or, min/max, sorted/reversed, ...), never a constant / index / slice /
    `/`->`//` change (RISKY), and
  * all equally good candidates for that function behave identically on
    random inputs shaped like the test's arguments (differential probe).
Otherwise the confirmed part is kept and the task goes to the LLM one-shot.
"""
import ast
import copy
import json
import os
import random
import re
import subprocess
import sys

from ..fsutil import atomic_write
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
SAFE, RISKY = "safe", "risky"
SAFE_BIN = {(ast.Add, ast.Sub), (ast.Sub, ast.Add), (ast.Add, ast.Mult), (ast.Mult, ast.Add), (ast.Mult, ast.Div),
            (ast.Div, ast.Mult)}
SAFE_CMP = {(ast.Lt, ast.Gt), (ast.Gt, ast.Lt), (ast.LtE, ast.GtE), (ast.GtE, ast.LtE), (ast.Eq, ast.NotEq),
            (ast.NotEq, ast.Eq), (ast.In, ast.NotIn), (ast.NotIn, ast.In), (ast.Is, ast.IsNot), (ast.IsNot, ast.Is)}
SAFE_ATTR = {("upper", "lower"), ("lower", "upper"), ("startswith", "endswith"), ("endswith", "startswith"),
             ("lstrip", "rstrip"), ("rstrip", "lstrip"), ("find", "rfind"), ("rfind", "find"),
             ("isupper", "islower"), ("islower", "isupper")}
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
    """Yield (mutator, kind): mutator changes `node` in place (on a deep copy)
    or is an UNWRAP_* marker; kind is SAFE (pure inverse flip) or RISKY."""
    if isinstance(node, (ast.BinOp, ast.AugAssign)):
        # an operator flip next to a literal (`+ 1` -> `* 1`, `n - 1` -> `n + 1`) is a
        # constant/index tweak in disguise, not a pure inverse flip
        lit = any(isinstance(o, ast.Constant) for o in (
            (node.left, node.right) if isinstance(node, ast.BinOp) else (node.value,)))
        for alt in BIN_ALT.get(type(node.op), []):
            yield (lambda n, alt=alt: setattr(n, "op", alt())), \
                SAFE if (type(node.op), alt) in SAFE_BIN and not lit else RISKY
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod)):
        def swap(n):
            n.left, n.right = n.right, n.left
        yield swap, RISKY
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            for alt in CMP_ALT.get(type(op), []):
                def f(n, i=i, alt=alt):
                    n.ops[i] = alt()
                yield f, SAFE if (type(op), alt) in SAFE_CMP else RISKY
    if isinstance(node, ast.BoolOp):
        yield (lambda n: setattr(n, "op", ast.Or() if isinstance(n.op, ast.And) else ast.And())), SAFE
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            yield (lambda n: setattr(n, "value", not n.value)), SAFE
        elif isinstance(v, int):
            for alt in dict.fromkeys([v + 1, v - 1, 0, 1, -v]):
                if alt != v:
                    yield (lambda n, alt=alt: setattr(n, "value", alt)), RISKY
        elif isinstance(v, float):
            for alt in (0.0, 1.0, v + 1, v - 1):
                if alt != v:
                    yield (lambda n, alt=alt: setattr(n, "value", alt)), RISKY
        elif isinstance(v, str) and v in ("", " ", ",", ", "):
            for alt in ("", " ", ",", ", "):
                if alt != v:
                    yield (lambda n, alt=alt: setattr(n, "value", alt)), RISKY
    if isinstance(node, ast.Name) and node.id in NAME_ALT:
        for alt in NAME_ALT[node.id]:
            yield (lambda n, alt=alt: setattr(n, "id", alt)), SAFE
    if isinstance(node, ast.Name) and node.id in ("True", "False"):
        yield (lambda n: setattr(n, "id", "False" if n.id == "True" else "True")), SAFE
    if isinstance(node, ast.Attribute) and node.attr in ATTR_ALT:
        for alt in ATTR_ALT[node.attr]:
            yield (lambda n, alt=alt: setattr(n, "attr", alt)), SAFE if (node.attr, alt) in SAFE_ATTR else RISKY
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "sorted":
            kws = [k for k in node.keywords if k.arg == "reverse"]
            if not kws:
                yield (lambda n: n.keywords.append(ast.keyword(arg="reverse", value=ast.Constant(True)))), SAFE
        for i, kw in enumerate(node.keywords):
            if kw.arg == "reverse" and isinstance(kw.value, ast.Constant):
                yield (lambda n, i=i: n.keywords.pop(i)), SAFE
        if isinstance(fn, ast.Attribute) and node.args and fn.attr in ("split", "strip", "lstrip", "rstrip", "pop"):
            yield (lambda n: n.args.clear()), RISKY  # s.split(' ') -> s.split()
        if isinstance(fn, ast.Name) and fn.id == "range" and node.args:
            def plus1(n):
                n.args[-1 if len(n.args) < 3 else 1] = ast.BinOp(
                    n.args[-1 if len(n.args) < 3 else 1], ast.Add(), ast.Constant(1))
            yield plus1, RISKY

            def minus1(n):
                n.args[-1 if len(n.args) < 3 else 1] = ast.BinOp(
                    n.args[-1 if len(n.args) < 3 else 1], ast.Sub(), ast.Constant(1))
            yield minus1, RISKY
            if len(node.args) == 1:
                yield (lambda n: n.args.insert(0, ast.Constant(1))), RISKY
    if isinstance(node, ast.BinOp) and isinstance(node.right, ast.Constant) and node.right.value in (1, -1) \
            and isinstance(node.op, (ast.Add, ast.Sub)):
        yield "UNWRAP_LEFT", RISKY
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
        yield "UNWRAP_OPERAND", RISKY
    if isinstance(node, ast.Slice):
        for attr in ("lower", "upper"):
            b = getattr(node, attr)
            if b is not None:
                yield (lambda n, attr=attr: setattr(n, attr, ast.BinOp(getattr(n, attr), ast.Add(),
                                                                       ast.Constant(1)))), RISKY
                yield (lambda n, attr=attr: setattr(n, attr, ast.BinOp(getattr(n, attr), ast.Sub(),
                                                                       ast.Constant(1)))), RISKY
                yield (lambda n, attr=attr: setattr(n, attr, None)), RISKY
        if node.step is not None:
            yield (lambda n: setattr(n, "step", None)), RISKY
        else:
            yield (lambda n: setattr(n, "step", ast.UnaryOp(ast.USub(), ast.Constant(1)))), RISKY
    if isinstance(node, ast.Return) and node.value is not None:
        yield (lambda n: setattr(n, "value", ast.UnaryOp(ast.Not(), n.value))), RISKY


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
    """All single-site mutants of a FunctionDef: ordered list of (source, kind)."""
    sites = list(ast.walk(func_node))
    out = {}
    for idx, node in enumerate(sites):
        for mut, kind in _mutants_of_node(node):
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
            if src not in out or (kind == SAFE and out[src] == RISKY):
                out[src] = kind
    return list(out.items())


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


def _second_order(src, funcs, failing, cap=6000):
    """Candidates with two simultaneous mutations inside one implicated function
    (e.g. clamp: min(lo, max(x, hi)) -> max(lo, min(x, hi)))."""
    fail_text = " ".join(failing).lower()
    implicated = [f for f in funcs if f.name.lower() in fail_text] or funcs
    cands, meta, kinds = [], [], []
    for f in implicated:
        seen = set()
        for m1, k1 in function_mutants(f):
            f1 = ast.parse(m1).body[0]
            for m2, k2 in function_mutants(f1):
                if m2 in seen:
                    continue
                seen.add(m2)
                cands.append(_replace_function(src, f, m2))
                meta.append(f.name)
                kinds.append(SAFE if k1 == k2 == SAFE else RISKY)
                if len(cands) >= cap:
                    break
    return cands, meta, kinds


def _rank(workspace, test_file, module, cands, failing, batch=1500):
    """[(gain, index, remaining failures)] for candidates that fix something
    and break nothing."""
    scored = []
    for start in range(0, len(cands), batch):
        for j, r in enumerate(_evaluate(workspace, test_file, module, cands[start:start + batch])):
            fset = set(r.get("failed", []))
            if "<import>" in fset or not fset <= failing:
                continue  # breaks something that used to work
            gain = len(failing) - len(fset)
            if gain > 0:
                scored.append((gain, start + j, frozenset(fset)))
    return scored


# ------------------------------------------------------------------ differential probe
def _call_templates(test_src, fname):
    """Literal positional-argument lists of every call to `fname` in the tests."""
    out = []
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.keywords:
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name != fname:
            continue
        try:
            out.append([ast.literal_eval(a) for a in node.args])
        except Exception:  # noqa: BLE001 - non-literal argument
            continue
    return out


_LETTERS = "abcdeiouxyzABCDE"


def _rand_like(v, rng):
    if isinstance(v, bool):
        return rng.random() < 0.5
    if isinstance(v, int):
        return rng.randint(-12, 12) if v < 0 else rng.randint(0, 12)
    if isinstance(v, float):
        return round(rng.uniform(-10, 10), 2)
    if isinstance(v, str):
        if " " in v:
            return " ".join("".join(rng.choice(_LETTERS) for _ in range(rng.randint(1, 4)))
                            for _ in range(rng.randint(1, 5)))
        return "".join(rng.choice(_LETTERS) for _ in range(rng.randint(0, 7)))
    if isinstance(v, (list, tuple)):
        proto = v[0] if v else 0
        items = [_rand_like(proto, rng) for _ in range(rng.randint(1, 8))]
        return items if isinstance(v, list) else tuple(items)
    return v


def probe_inputs(test_src, fname, n=60, seed=7):
    templates = _call_templates(test_src, fname)
    if not templates:
        return []
    rng = random.Random(seed)
    inputs = [list(t) for t in templates]
    for i in range(n):
        t = templates[i % len(templates)]
        inputs.append([_rand_like(a, rng) for a in t])
    return [list(a) if isinstance(a, tuple) else a for a in inputs]


def probe(workspace, module, fname, sources, inputs, timeout=60):
    """Behaviour of `fname` in each module source on the given inputs."""
    req = json.dumps({"mode": "probe", "workspace": workspace, "module": module, "function": fname,
                      "candidates": sources, "inputs": inputs})
    try:
        p = subprocess.run([sys.executable, "-I", RUNNER], input=req, capture_output=True, text=True,
                           timeout=timeout, cwd=workspace)
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


# strict by default: a disputed repair costs ~100 tokens on the LLM, a wrong one costs the task
DIFF_POLICY = os.environ.get("CODING_DIFF_POLICY", "strict")


def _winner_trusted(beh):
    """beh[0] is the winner's behaviour, beh[1:] its rivals'. 'strict': all must
    agree. 'plurality': the winner's behaviour class must be strictly the most
    common one -- a true fix usually has several equivalent spellings
    (`% 2 != 1` == `% 2 == 0`), a coincidental pass (`% 3 == 1`) is a loner."""
    if all(b == beh[0] for b in beh[1:]):
        return True
    if DIFF_POLICY == "strict":
        return False
    classes = {}
    for b in beh:
        key = json.dumps(b)
        classes[key] = classes.get(key, 0) + 1
    win = classes[json.dumps(beh[0])]
    return all(win > n for k, n in classes.items() if k != json.dumps(beh[0]))


def _func_src(module_src, name):
    for n in ast.parse(module_src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.unparse(n)
    return None


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

    test_src = open(tpath, encoding="utf-8").read()
    history = []
    escalate = None
    diff_info = None
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
        cands, meta, kinds = [], [], []
        for f in funcs:
            for msrc, kind in function_mutants(f):
                cands.append(_replace_function(src, f, msrc))
                meta.append(f.name)
                kinds.append(kind)
        scored = _rank(workspace, test_file, module, cands, failing)
        if not scored:
            cands, meta, kinds = _second_order(src, funcs, failing)
            scored = _rank(workspace, test_file, module, cands, failing)
            if not scored:
                break
        top = max(g for g, _, _ in scored)
        best = [(i, fset) for g, i, fset in scored if g == top]
        safe = [(i, fset) for i, fset in best if kinds[i] == SAFE]
        if not safe:
            escalate = f"only non-inverse repairs fit {meta[best[0][0]]}()"
            break
        win, win_fset = safe[0]
        rivals = [i for i, fset in best if meta[i] == meta[win] and fset == win_fset]
        if len(rivals) > 1:
            inputs = probe_inputs(test_src, meta[win])
            beh = probe(workspace, module, meta[win], [cands[i] for i in rivals], inputs) if inputs else None
            if beh is None or not _winner_trusted(beh):
                escalate = f"{len(rivals)} candidate repairs of {meta[win]}() behave differently"
                diff_info = [_func_src(cands[i], meta[win]) for i in rivals]
                break
        src, failing = cands[win], set(win_fset)
        history.append(meta[win])
        if not failing:
            break

    if failing:
        if history:  # keep the confirmed inverse flips; the LLM only sees what is still broken
            atomic_write(spath, src)
        return SolveResult.unknown(escalate or f"mutation search left {len(failing)} failing tests",
                                   failing=sorted(failing), edits=history, rivals=diff_info)
    atomic_write(spath, src)
    ok, out = run_tests(workspace, test_file)
    if not ok:
        atomic_write(spath, original)
        return SolveResult.unknown("candidate passed runner but not the real test command", output=out[-800:])
    return SolveResult.solved("DONE", f"fixed by {len(history)} mutation(s) in {sorted(set(history))}",
                              edits=history)
