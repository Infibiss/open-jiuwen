"""Score a tasks directory after a run (needs the `expected` field).

    python tools/evaluate.py /tmp/train [--results results.json]

Rules mirror the regulations: a task scores its reward only if answer.txt is
correct (coding: DONE + test command passes). For infeasible tasks
(expected == "__INFEASIBLE__") we additionally report whether the harness
correctly abstained (no answer.txt) - this earns no points.
"""
import json
import os
import subprocess
import sys

INF = "__INFEASIBLE__"


def check(task_dir, task):
    ans_path = os.path.join(task_dir, "answer.txt")
    has = os.path.isfile(ans_path)
    ans = open(ans_path, encoding="utf-8").read().strip() if has else None
    exp = task.get("expected")
    if exp == INF:
        return False, ("abstained" if not has else f"answered infeasible: {ans!r}")
    if not has:
        return False, "no answer"
    if task.get("check") == "unittest" or exp == "tests_pass":
        ws = os.path.join(task_dir, "workspace")
        p = subprocess.run([sys.executable, "test_mylib.py"], cwd=ws, capture_output=True, text=True, timeout=60)
        ok = p.returncode == 0 and ans == "DONE"
        return ok, "tests pass" if ok else f"tests rc={p.returncode} ans={ans!r}"
    if task.get("check") == "exact_int":
        try:
            ok = int(ans) == int(exp)
        except ValueError:
            ok = False
    else:
        ok = ans == str(exp).strip()
    return ok, f"got {ans!r} expected {exp!r}"


def main():
    root = sys.argv[1]
    total = earned = 0
    by_type = {}
    rows = []
    for d in sorted(os.listdir(root)):
        tj = os.path.join(root, d, "task.json")
        if not os.path.isfile(tj):
            continue
        task = json.load(open(tj, encoding="utf-8"))
        ok, why = check(os.path.join(root, d), task)
        r = task.get("reward", 1)
        feasible = task.get("expected") != INF
        total += r if feasible else 0
        earned += r if ok else 0
        bt = by_type.setdefault(task["task_type"], [0, 0, 0])
        bt[0] += ok
        bt[1] += feasible
        bt[2] += (not feasible) and why == "abstained"
        rows.append((d, ok, why))
    for d, ok, why in rows:
        if not ok:
            print(f"  {'--' if why == 'abstained' else 'XX'} {d}: {why}")
    for t, (ok, feas, abst) in sorted(by_type.items()):
        print(f"{t:15s} solved {ok}/{feas} feasible, abstained on infeasible: {abst}")
    print(f"SCORE {earned}/{total} achievable points")


if __name__ == "__main__":
    main()
