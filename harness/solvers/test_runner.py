"""Sandboxed batch evaluator for candidate module sources (run as a subprocess).

stdin:  {"workspace": dir, "test_file": "test_mylib.py", "module": "mylib",
         "candidates": [source, ...]}
stdout: [{"ok": bool, "failed": [test ids]}, ...]  (one entry per candidate)

Every candidate is executed as a fresh module injected into sys.modules and
the (unchanged) test file is re-executed against it; a per-candidate alarm
protects against infinite loops introduced by a mutation.
"""
import io
import json
import os
import signal
import sys
import types
import unittest


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _collect_ids(test_ns):
    ids = []
    for name, obj in sorted(test_ns.items()):
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            for t in unittest.defaultTestLoader.getTestCaseNames(obj):
                ids.append(f"{name}.{t}")
        elif name.startswith("test") and callable(obj) and isinstance(obj, types.FunctionType):
            ids.append(name)
    return ids


def evaluate(test_code, test_path, module, src, per_test_timeout=2.0):
    for k in list(sys.modules):
        if k == module:
            del sys.modules[k]
    mod = types.ModuleType(module)
    mod.__file__ = os.path.join(os.getcwd(), module + ".py")
    signal.setitimer(signal.ITIMER_REAL, per_test_timeout * 3)
    try:
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        sys.modules[module] = mod
        ns = {"__name__": "candidate_tests", "__file__": test_path}
        exec(test_code, ns)
    except BaseException as e:  # noqa: BLE001 - any failure means the candidate is broken
        signal.setitimer(signal.ITIMER_REAL, 0)
        return {"ok": False, "failed": ["<import>"], "error": type(e).__name__}
    signal.setitimer(signal.ITIMER_REAL, 0)
    failed = []
    ids = _collect_ids(ns)
    for tid in ids:
        ok = False
        signal.setitimer(signal.ITIMER_REAL, per_test_timeout)
        try:
            if "." in tid:
                cls, meth = tid.split(".", 1)
                suite = unittest.TestSuite([ns[cls](meth)])
                res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
                ok = res.wasSuccessful() and res.testsRun == 1
            else:
                ns[tid]()
                ok = True
        except BaseException:  # noqa: BLE001
            ok = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if not ok:
            failed.append(tid)
    return {"ok": not failed and bool(ids), "failed": failed, "n": len(ids)}


def main():
    req = json.load(sys.stdin)
    ws = req["workspace"]
    os.chdir(ws)
    sys.path.insert(0, ws)
    signal.signal(signal.SIGALRM, _alarm)
    test_path = os.path.join(ws, req["test_file"])
    with open(test_path, encoding="utf-8") as f:
        test_code = compile(f.read(), test_path, "exec")
    out = []
    real_stdout = sys.stdout
    for src in req["candidates"]:
        sys.stdout = io.StringIO()  # silence prints from candidates/tests
        try:
            out.append(evaluate(test_code, test_path, req["module"], src))
        except BaseException as e:  # noqa: BLE001
            out.append({"ok": False, "failed": ["<crash>"], "error": type(e).__name__})
        finally:
            sys.stdout = real_stdout
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
