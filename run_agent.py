"""Entrypoint: python run_agent.py --tasks-dir /data/tasks [--out results.json]

Environment: LLM_BASE_URL, LLM_API_KEY, BUDGET (default 100000), OUT, TASKS.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Budget-aware openJiuwen harness")
    ap.add_argument("--tasks-dir", "--tasks", dest="tasks_dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", "--output", "--out-file", dest="out", default=None,
                    help="solutions/results JSON (default: $OUT or /tmp/results.json)")
    ap.add_argument("--predict-only", action="store_true",
                    help="only forecast the token cost of every task (no LLM calls, task files untouched)")
    ap.add_argument("positional", nargs="*", help="optional: TASKS_DIR [OUT]")
    args = ap.parse_args()
    if args.positional:
        args.tasks_dir = args.positional[0]
        if len(args.positional) > 1 and not args.out:
            args.out = args.positional[1]

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
