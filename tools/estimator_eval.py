"""Offline check of the token estimator against a recorded run.

    python tools/estimator_eval.py <tasks_dir> <results.json>

1. Whole-task (the honest metric): the forecast the harness recorded *before*
   starting each LLM task (`predicted_cost.expected_total`, including the
   expected escalations) vs the task's total recorded spend.
2. First call only: re-prices every task with the current priors (no online
   correction) for the strategy that was actually used.
Run it on a seed the priors were not fitted on.
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import llm_solvers  # noqa: E402
from harness.estimator import Estimator  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402


def main():
    tasks = {t.id: t for t in load_tasks(sys.argv[1])}
    res = json.load(open(sys.argv[2], encoding="utf-8"))
    whole = [(r["task_type"], r["predicted_cost"]["expected_total"], r["actual_cost"]) for r in res["tasks"]
             if r.get("predicted_cost", {}).get("expected_total") and r["actual_cost"]]
    if whole:
        mae = st.mean(abs(p - a) for _, p, a in whole)
        print(f"WHOLE TASK n={len(whole)} mean actual={st.mean(a for *_, a in whole):.0f} "
              f"mean predicted={st.mean(p for _, p, _ in whole):.0f} MAE={mae:.0f} "
              f"median abs err={st.median(abs(p - a) for _, p, a in whole):.0f} "
              f"underestimates={sum(p < a for _, p, a in whole) / len(whole):.0%} "
              f"total predicted={sum(p for _, p, _ in whole):.0f} total actual={sum(a for *_, a in whole):.0f}")
    est = Estimator()
    rows = []
    for r in res["tasks"]:
        t = tasks.get(r["task_id"])
        if t is None or not r["actual_cost"] or r["strategy"] not in ("oneshot", "react"):
            continue
        if r["strategy"] == "oneshot":
            prompt = llm_solvers.build(t)
            if prompt is None:
                continue
            p = est.predict(t, "oneshot", prompt.text, prompt.target_text)["cost"]
        else:
            p = est.predict(t, "react")["cost"]
        rows.append((t.type, r["strategy"], p, r["actual_cost"]))
    if not rows:
        print("nothing to compare")
        return
    for ty in sorted({(a, b) for a, b, _, _ in rows}):
        sel = [(p, a) for t, s, p, a in rows if (t, s) == ty]
        print(f"{ty[0]:14s} {ty[1]:8s} n={len(sel):3d} pred={st.mean(p for p, _ in sel):6.0f} "
              f"actual={st.mean(a for _, a in sel):6.0f} MAE={st.mean(abs(p - a) for p, a in sel):5.0f}")
    mae = st.mean(abs(p - a) for _, _, p, a in rows)
    rel = st.mean(abs(p - a) / a for _, _, p, a in rows)
    under = sum(p < a for _, _, p, a in rows) / len(rows)
    print(f"ALL n={len(rows)} MAE={mae:.0f} tokens, mean rel. error={rel:.0%}, underestimates={under:.0%}")


if __name__ == "__main__":
    main()
