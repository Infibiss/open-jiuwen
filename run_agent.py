"""Entrypoint: python run_agent.py --tasks-dir /data/tasks [--out results.json]

Environment: LLM_BASE_URL, LLM_API_KEY, BUDGET (default 100000), OUT, TASKS.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Budget-aware openJiuwen harness")
    ap.add_argument("--tasks-dir", "--tasks", "--tasks_dir", dest="tasks_dir", default=None)
    ap.add_argument("--out", "--output", "--out-file", "--output-file", "--solutions", dest="out", default=None,
                    help="solutions/results JSON (default: $OUT or /tmp/results.json)")
    ap.add_argument("--predict-only", action="store_true",
                    help="only forecast the token cost of every task (no LLM calls, task files untouched)")
    ap.add_argument("positional", nargs="*", help="optional: TASKS_DIR [OUT]")
    args, unknown = ap.parse_known_args()
    if unknown:  # never die on a flag the jury's launcher adds
        print(f"ignoring unknown arguments: {unknown}", file=sys.stderr)
    # Positionals are only trusted when they look right: the value of an unknown
    # flag (e.g. "--new-flag x") must not become the tasks dir or the output file.
    pos = args.positional
    if args.tasks_dir is None:
        dirs = [p for p in pos if os.path.isdir(os.path.join(p, "")) and not p.endswith(".json")]
        args.tasks_dir = dirs[0] if dirs else os.environ.get("TASKS", "/data/tasks")
    if not args.out:
        files = [p for p in pos if p != args.tasks_dir and not os.path.isdir(p)
                 and (p.lower().endswith((".json", ".jsonl", ".txt")) or os.sep in p)]
        args.out = files[0] if files else None

    from harness.scheduler import Harness
    from harness.tasks import resolve_out_path

    if not os.path.isdir(args.tasks_dir):
        print(f"tasks dir not found: {args.tasks_dir}", file=sys.stderr)
        return 2
    h = Harness(args.tasks_dir, resolve_out_path(args.out))
    if args.predict_only:
        h.predict_only()
    else:
        h.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
