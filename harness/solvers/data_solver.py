"""Deterministic solver for the sales.csv aggregation questions.

Supported (everything else is escalated to the LLM):
  * totals: units sold (sum of `units`), number of orders (row count),
    revenue (sum of units*price)
  * "which <column> sold the most/least units|orders|revenue" (ties -> LLM)
  * largest/smallest units in a single order, average units/price
    (integer or explicit rounding only), number of distinct values
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
    alts, ors = {}, {}  # column -> accepted values ("or") / end of its last quoted value
    blank = []  # spans of connector words consumed by the filters ("or", "outside")

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
        if any(c == col for c, _ in filters):
            # "region 'a' or region 'b'" (only "or": "and" on one column is ambiguous)
            prev = [i for i, (c, _) in enumerate(filters) if c == col]
            between = q[ors[col]:m.start()] if col in ors else ""
            if len(prev) == 1 and re.fullmatch(r"\s*or\s+(?:in\s+)?", between, re.I) and alts.get(col):
                alts[col].add(val)
                blank.append((ors[col], m.start()))
                ors[col] = m.end()
                continue
            return SolveResult.unknown(f"two filters on column {col!r}")
        neg = re.search(r"\b(?:outside(?: of)?|except(?: in)?|excluding|not in|other than)\s+(?:the\s+)?$",
                        q[:m.start()], re.I)
        if neg:  # "sold outside region 'north'"
            blank.append((neg.start(), neg.end()))
            filters.append((col, (lambda c, v: (lambda r: r[c] != v))(col, val)))
            alts[col], ors[col] = None, m.end()  # no "or" after a negation
            continue
        alts[col], ors[col] = {val}, m.end()
        filters.append((col, (lambda c, vs: (lambda r: r[c] in vs))(col, alts[col])))
    for a, b in sorted(blank, reverse=True):
        q = q[:a] + " " * (b - a) + q[b:]
    q = re.sub(r"\b(\w+) ['\"]([^'\"]+)['\"]", " ", q)
    if re.search(r"\b(?:year|years) \d{4}\b|\bin \d{4}\b", q, re.I) and not any(
            "year" in c or "date" in c for c in header):
        return SolveResult.infeasible("question filters by year but the csv has no year or date column")

    # numeric conditions: "had more than 12 units", "at a price above 46", "with at least 3 units"
    for m in list(_NUMCMP.finditer(q)):
        col = _col(m["c1"] or m["c2"] or "", header) if (m["c1"] or m["c2"]) else None
        if col is None or (m["c1"] and m["c2"] and _col(m["c2"], header) != col):
            return SolveResult.unknown(f"comparison on an unknown column in {question!r}")
        op, t = CMP[m["op"].lower()], float(m["n"])
        filters.append((col, (lambda c, op, t: (lambda r: op(_to_num(r[c]), t)))(col, op, t)))
        q = q.replace(m[0], " ", 1)

    # month filters
    if "month" not in header and re.search(r"\bmonths?\b", q, re.I):
        return SolveResult.unknown("question filters by month but csv has no month column")
    if "month" in header:
        mr = re.search(r"\bmonths? (\d+) ?(?:through|to|-) ?(\d+)(?: inclusive)?", q, re.I)
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

    sel = [r for r in rows if all(p(r) for _, p in filters)]
    try:
        res = _answer(q.lower(), header, rows, sel)
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return SolveResult.unknown(f"bad numeric data: {e}")
    if isinstance(res, SolveResult):
        if res.status == "unknown":
            res.reason += f" in {question!r}"
        return res
    kind, ans = res
    return SolveResult.solved(_fmt(ans), f"{kind} with {len(filters)} filters over {len(sel)} rows",
                              rows=len(rows), selected=len(sel))


def _leftover(ql, extra=()):
    """Words (and numbers: a year, a second month, a percentage...) the template did not consume."""
    allowed = ALLOWED | set(extra)
    return [w for w in re.findall(r"[a-z]+|\d+", ql) if w not in allowed]


def _col(word, header):
    word = word.lower()
    for c in (word, word[:-1] if word.endswith("s") else None, word[:-2] if word.endswith("es") else None):
        if c and c in header:
            return c
    return None


CMP = {"more than": lambda a, b: a > b, "greater than": lambda a, b: a > b, "over": lambda a, b: a > b,
       "above": lambda a, b: a > b, "exceeding": lambda a, b: a > b, "higher than": lambda a, b: a > b,
       "less than": lambda a, b: a < b, "fewer than": lambda a, b: a < b, "under": lambda a, b: a < b,
       "below": lambda a, b: a < b, "lower than": lambda a, b: a < b,
       "at least": lambda a, b: a >= b, "no less than": lambda a, b: a >= b, "no fewer than": lambda a, b: a >= b,
       "at most": lambda a, b: a <= b, "no more than": lambda a, b: a <= b}
_NUMCMP = re.compile(
    r"\b(?:(?:had|has|have|with|having|at|for)\s+)?(?:(?:a|an)\s+)?(?:(?P<c1>units?|price|prices)\s+(?:of\s+)?)?"
    r"(?P<op>" + "|".join(sorted(CMP, key=len, reverse=True)) + r")\s+(?P<n>\d+(?:\.\d+)?)"
    r"(?:\s+(?P<c2>units?|price))?\b", re.I)

MAX_W = {"most", "highest", "largest", "greatest", "biggest", "maximum", "max", "top"}
MIN_W = {"least", "lowest", "smallest", "fewest", "minimum", "min"}
EXT = "|".join(sorted(MAX_W | MIN_W))
GROUP_EXTRA = ("which", "had", "has", "have", "recorded", "generated", "got", "saw", "most", "least", "highest",
               "lowest", "largest", "smallest", "fewest", "greatest", "biggest", "answer", "name", "just", "only",
               "its", "value", "sold")


def _metric_value(metric, rs):
    if metric == "units":
        return sum(_to_num(r["units"]) for r in rs)
    if metric == "orders":
        return len(rs)
    return sum(_to_num(r["units"]) * _to_num(r["price"]) for r in rs)


def _answer(ql, header, rows, sel):
    """Returns (kind, value) or a SolveResult (unknown / infeasible)."""
    # 1. arg-extreme over a group: "Which region sold the most units ...", "In which month were the most units ..."
    m = re.search(rf"\b(?:in )?which (\w+) (?:\w+ )?(?:the )?({EXT}) (?:total )?(?:number of )?(units|revenue|orders)\b", ql)
    if m:
        col = _col(m[1], header)
        if col is None:
            return SolveResult.unknown(f"group column {m[1]!r} not in csv")
        metric = m[3]
        if metric == "revenue" and ("units" not in header or "price" not in header):
            return SolveResult.unknown("revenue without units/price")
        rest = ql[:m.start()] + " " + ql[m.end():]
        rest = re.sub(r"\(?\s*units\s*[*x×]\s*price\s*\)?", " ", rest)
        left = _leftover(rest, GROUP_EXTRA + (m[1],))
        if left:
            return SolveResult.unknown(f"unrecognised words {left}")
        groups = {}
        for r in sel:
            groups.setdefault(r[col], []).append(r)
        if not groups:
            return SolveResult.unknown("no matching rows")
        if m[2] in MIN_W and len(groups) < len({r[col] for r in rows}):
            return SolveResult.unknown("some groups have no rows (minimum ambiguous)")
        tot = sorted(((_metric_value(metric, rs), g) for g, rs in groups.items()), reverse=m[2] in MAX_W)
        if len(tot) > 1 and tot[0][0] == tot[1][0]:
            return SolveResult.unknown("tie for the extreme group")
        return f"argext {metric} by {col}", tot[0][1]

    # 2. extreme of a single order: "largest number of units in a single order"
    m = re.search(rf"\b({EXT}) (?:number of )?units (?:in|of|for|sold in|within) (?:a|one|any) (?:single )?order\b", ql)
    if m:
        rest = ql[:m.start()] + " " + ql[m.end():]
        left = _leftover(rest, ("single", "one"))
        if left:
            return SolveResult.unknown(f"unrecognised words {left}")
        if not sel:
            return SolveResult.infeasible("no order matches the filters")
        vals = [_to_num(r["units"]) for r in sel]
        return "single-order extreme", (max(vals) if m[1] in MAX_W else min(vals))

    # 3. average units / price
    m = re.search(r"\baverage (?:number of )?(units|price)(?: per order| of an order| in an order| per unit)?\b", ql)
    if m:
        rest = ql[:m.start()] + " " + ql[m.end():]
        rnd = re.search(r"\(?\s*(?:rounded|round) (down|to the nearest (?:integer|whole number))\s*\)?", rest)
        if rnd:
            rest = rest[:rnd.start()] + " " + rest[rnd.end():]
        left = _leftover(rest)
        if left:
            return SolveResult.unknown(f"unrecognised words {left}")
        if not sel:
            return SolveResult.infeasible("average over no rows")
        avg = sum(_to_num(r[m[1]]) for r in sel) / len(sel)
        if rnd:
            avg = int(avg // 1) if rnd[1] == "down" else round(avg)
        elif not float(avg).is_integer():
            return SolveResult.unknown("non-integer average without rounding rule")
        return f"average {m[1]}", avg

    # 4. distinct count: "How many different products ..."
    m = re.match(r"\s*how many (?:different|distinct|unique) (\w+)\b", ql)
    if m:
        col = _col(m[1], header)
        if col is None:
            return SolveResult.unknown(f"column {m[1]!r} not in csv")
        left = _leftover(ql[m.end():], ("appear", "appears", "occur", "occurs", "exist", "exists", "had", "have"))
        if left:
            return SolveResult.unknown(f"unrecognised words {left}")
        return f"distinct {col}", len({r[col] for r in sel})

    # 5. totals: units / orders / revenue
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
        return SolveResult.unknown("metric not recognised")
    left = _leftover(ql)
    if left:
        return SolveResult.unknown(f"unrecognised words {left}")
    return metric, _metric_value(metric, sel)
