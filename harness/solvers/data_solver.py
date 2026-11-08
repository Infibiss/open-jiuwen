"""Deterministic solver for the sales.csv aggregation questions.

Supported (everything else is escalated to the LLM):
  * metric: units sold (sum of `units`), number of orders (row count),
    revenue (sum of units*price)
  * filters: any categorical column given as  <column> 'value'
             (product 'widget', region 'north'), month ranges
             ("months 7 through 8 inclusive", "month 5").
The question is consumed token by token; any word outside a small
whitelist means the phrasing is unfamiliar -> UNKNOWN.
"""
import csv
import os
import re

from .base import SolveResult

_TAIL = re.compile(r"\s*(Use bash|Use python|Use the|Compute it|Work only inside|When you are done|Write ONLY)\b.*$", re.I | re.S)
_FILE = re.compile(r"The file (\S+?\.csv) has columns ([\w,]+)(?: \((\d+) rows?\))?\.?", re.I)

ALLOWED = set("""
how many what was were is are the a an of in for from during total number sold
units unit orders order product products region regions month months through to
inclusive and with between by there all placed sales item items sell overall
""".split())


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_num(v):
    v = str(v).strip()
    try:
        return int(v)
    except ValueError:
        return float(v)


def _fmt(x):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x)


def solve(task, workspace):
    instr = task.get("instruction", "")
    fm = _FILE.search(instr)
    if not fm:
        return SolveResult.unknown("csv description not found")
    fname = fm[1]
    path = os.path.join(workspace or "", fname)
    if not os.path.isfile(path):
        return SolveResult.infeasible(f"data file {fname} is missing")
    try:
        rows = _read_csv(path)
    except Exception as e:  # malformed file -> let the LLM look at it
        return SolveResult.unknown(f"csv read error: {e}")
    if not rows:
        return SolveResult.unknown("empty csv")
    header = list(rows[0].keys())

    question = _TAIL.sub("", instr[fm.end():]).strip()
    q = question
    filters = []  # (column, predicate)

    # categorical filters:  product 'widget' / region "north"
    for m in list(re.finditer(r"\b(\w+) ['\"]([^'\"]+)['\"]", q)):
        col, val = m[1].lower(), m[2]
        if col.endswith("s") and col[:-1] in header:
            col = col[:-1]
        if col not in header:
            return SolveResult.unknown(f"quoted value for unknown column {col!r}")
        values = {r[col] for r in rows}
        if val not in values:
            ci = {v.lower(): v for v in values}
            if val.lower() in ci:
                val = ci[val.lower()]
            else:
                return SolveResult.infeasible(f"{col} {val!r} does not occur in {fname}")
        filters.append((col, (lambda c, v: (lambda r: r[c] == v))(col, val)))
    q = re.sub(r"\b(\w+) ['\"]([^'\"]+)['\"]", " ", q)

    # month filters
    if "month" not in header and re.search(r"\bmonths?\b", q, re.I):
        return SolveResult.unknown("question filters by month but csv has no month column")
    if "month" in header:
        mr = re.search(r"\bmonths? (\d+) (?:through|to|-|and) (\d+)(?: inclusive)?", q, re.I)
        if mr:
            a, b = int(mr[1]), int(mr[2])
            if not (1 <= a <= 12 and 1 <= b <= 12) or a > b:
                return SolveResult.infeasible(f"invalid month range {a}..{b}")
            filters.append(("month", lambda r, a=a, b=b: a <= int(r["month"]) <= b))
            q = q[:mr.start()] + " " + q[mr.end():]
        else:
            ms = re.search(r"\bmonth (\d+)\b", q, re.I)
            if ms:
                k = int(ms[1])
                if not 1 <= k <= 12:
                    return SolveResult.infeasible(f"invalid month {k}")
                filters.append(("month", lambda r, k=k: int(r["month"]) == k))
                q = q[:ms.start()] + " " + q[ms.end():]

    ql = q.lower()
    if re.search(r"\b(revenue|value|amount|income|earn)", ql):
        if "units" not in header or "price" not in header:
            return SolveResult.unknown("revenue without units/price")
        metric = "revenue"
        ql = re.sub(r"\(?\s*units\s*[*x×]\s*price\s*\)?", " ", ql)
        ql = re.sub(r"\b(revenue|value|amount|income|earned|generated|in)\b", " ", ql)
    elif re.match(r"\s*how many units\b", ql) or re.search(r"\btotal (number of )?units\b", ql):
        metric = "units"
    elif re.match(r"\s*how many orders\b", ql) or re.search(r"\bnumber of orders\b", ql):
        metric = "orders"
    else:
        return SolveResult.unknown(f"metric not recognised: {question!r}")

    leftover = [w for w in re.findall(r"[a-z]+", ql) if w not in ALLOWED]
    if leftover:
        return SolveResult.unknown(f"unrecognised words {leftover} in {question!r}")

    sel = [r for r in rows if all(p(r) for _, p in filters)]
    try:
        if metric == "units":
            ans = sum(_to_num(r["units"]) for r in sel)
        elif metric == "orders":
            ans = len(sel)
        else:
            ans = sum(_to_num(r["units"]) * _to_num(r["price"]) for r in sel)
    except (KeyError, ValueError) as e:
        return SolveResult.unknown(f"bad numeric data: {e}")
    return SolveResult.solved(_fmt(ans), f"{metric} with {len(filters)} filters over {len(sel)} rows",
                              rows=len(rows), selected=len(sel))
