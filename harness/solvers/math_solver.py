"""Deterministic solver for the "warehouse stock" arithmetic word problems.

The problem text is split into sentences; every sentence must match one of
the known operation templates, otherwise the task is escalated to the LLM
(status UNKNOWN). Arithmetic mirrors what `python3 -c` would compute
(integers, floor division for "rounded down").
"""
import re

from .base import SolveResult

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a": 1, "an": 1, "once": 1, "twice": 2, "thrice": 3,
}
N = r"(\d[\d,]*|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
W = r"[a-z][a-z\- ]*?"  # entity name in the first sentence ("warehouse", "small shop")
U = r"(?P<u>[a-z]+)"    # ONE unit noun; must be the stock's unit (checked), so "percent", "deliveries" fail
C = r"[a-z]+"           # one container noun (warehouses, stores, ...)
TIME = r"(?:day|week|month|hour)"

# Instruction boilerplate that follows the actual problem statement.
_TAIL = re.compile(
    r"\s*(Compute it|Use (?:the )?bash|Use python|Work only inside|When you are done|Write ONLY)\b.*$",
    re.I | re.S,
)
_PREFIX = re.compile(r"^(?:then|next|after that|afterwards|finally|later)\s*,?\s*", re.I)

# quantities that are not given -> the exact answer cannot be computed
_VAGUE = re.compile(r"\b(some|several|a few|few|many|an unknown|unknown|a certain|certain number|an unspecified|"
                    r"unspecified|a number of|a lot of|a couple of|most|part of)\b", re.I)

_START = re.compile(rf"^(?:a|an|the) ({W}) (?:starts|begins|starts out|opens) with {N} ({W})$", re.I)
_QUESTION = re.compile(rf"^how many ({W}) does (?:it|the {W}) have (?:at the end|in the end|now|left)$", re.I)
SCALE = {"doubled": 2, "tripled": 3, "quadrupled": 4}
FRACTION = {"half": 2, "a half": 2, "one half": 2, "a third": 3, "one third": 3, "a quarter": 4, "one quarter": 4,
            "a fourth": 4, "one fourth": 4, "a fifth": 5, "one fifth": 5, "a tenth": 10, "one tenth": 10}
FR = r"(half|a half|one half|a third|one third|a quarter|one quarter|a fourth|one fourth|a fifth|one fifth|a tenth|one tenth)"
OUT = r"(?:ships|sells|sends out|gives away|donates|loses|discards|uses)"
DOWN = r"(?P<down>,? (?:\(rounded down\)|rounded down))?"


def _share_out(s, num, den, down):
    """Stock after giving away num/den of it; the amount is rounded down only when the text says so."""
    if not down and (s * num) % den:
        return False  # fractional amount without a rounding rule: leave it to the LLM
    return s - s * num // den


def _num(s):
    s = s.strip().lower().replace(",", "")
    if s.isdigit():
        return int(s)
    return NUM_WORDS[s]


def _ops():
    """(regex, fn(match, stock) -> new stock | None for division by zero).
    Every pattern is anchored at both ends, so any extra clause ("to each of
    three stores", "but returns half of them", "of its stock") falls through
    to the LLM. Group 'u' (if present) must name the stock's unit."""
    ops = []

    def add(pattern, fn):
        ops.append((re.compile("^" + pattern + "$", re.I), fn))

    # it ships 36 boxes per day for 3 days
    # both time words must match: "per day for 2 weeks" needs a conversion and goes to the LLM
    add(rf"it (?:ships|sells|sends out|uses|loses) {N} {U} (?:per|a|each|every) (?P<t>{TIME}) for {N} (?P=t)s?",
        lambda m, s: s - _num(m[1]) * _num(m[4]))
    # per day for 2 weeks: a week is always 7 days (months and hours stay with the LLM)
    add(rf"it (?:ships|sells|sends out|uses|loses) {N} {U} (?:per|a|each|every) day for {N} weeks?",
        lambda m, s: s - _num(m[1]) * 7 * _num(m[3]))
    # it receives 2 deliveries of 87 boxes each
    add(rf"it (?:receives|gets|takes in|accepts) {N} (?:deliveries|shipments|trucks|truckloads|batches|pallets|loads) "
        rf"of {N} {U}(?: each)?", lambda m, s: s + _num(m[1]) * _num(m[2]))
    # it receives 40 boxes / a delivery of 40 boxes
    add(rf"it (?:receives|gets|buys|takes in|accepts)(?: a (?:delivery|shipment) of)? {N} (?:more |new |extra )?{U}",
        lambda m, s: s + _num(m[1]))
    # it ships 40 boxes
    add(rf"it (?:ships|sells|sends out|gives away|donates|loses|discards|returns|uses) {N} {U}",
        lambda m, s: s - _num(m[1]))
    # 12 boxes are damaged and discarded
    add(rf"{N} {U} (?:are|were|get|got) (?:damaged|lost|stolen|discarded|removed|destroyed|broken)"
        rf"(?: and (?:discarded|removed|thrown away))?", lambda m, s: s - _num(m[1]))
    # the stock is split equally into 3 warehouses and it keeps one share (rounded down)
    add(rf"the stock is (?:split|divided) (?:equally|evenly) (?:into|among|between) {N} {C} and it keeps one "
        rf"(?:share|part|portion)(?: \(rounded down\)| rounded down)?",
        lambda m, s: None if _num(m[1]) == 0 else s // _num(m[1]))
    # the stock is doubled (by merging with a sister warehouse)
    add(rf"the stock is (doubled|tripled|quadrupled)(?: by merging with (?:a|another|its) sister {C})?",
        lambda m, s: s * SCALE[m[1].lower()])
    add(rf"the stock is multiplied by {N}", lambda m, s: s * _num(m[1]))
    add(r"the stock is halved(?: \(rounded down\)| rounded down)?", lambda m, s: s // 2)
    add(rf"the stock is divided by {N}(?: \(rounded down\)| rounded down)?",
        lambda m, s: None if _num(m[1]) == 0 else s // _num(m[1]))
    # it gives away a third of its stock (rounded down) / it ships half of its stock
    add(rf"it {OUT} {FR} of (?:its|the) (?:stock|inventory|{U}){DOWN}",
        lambda m, s: _share_out(s, 1, FRACTION[m[1].lower()], m["down"]))
    # it sells 25 percent of its stock, rounded down / 25% of its boxes
    add(rf"it {OUT} (\d+)(?: percent|%) of (?:its|the) (?:stock|inventory|{U}){DOWN}",
        lambda m, s: _share_out(s, int(m[1]), 100, m["down"]) if int(m[1]) <= 100 else False)
    # it ships 16 boxes to each of three stores
    add(rf"it {OUT} {N} {U} (?:to|for) each of {N} {C}", lambda m, s: s - _num(m[1]) * _num(m[3]))
    return ops


# operations whose size is not given -> no exact answer
MISSING = [re.compile("^" + p + "$", re.I) for p in (
    rf"it (?:receives|gets|takes in|accepts) {N} (?:deliveries|shipments|trucks|truckloads|batches|pallets|loads)",
    rf"it (?:ships|sells|sends out|uses|loses) {N} {U} (?:per|a|each|every) {TIME}",
    rf"it (?:ships|sells|sends out|uses|loses) {U} (?:per|a|each|every) {TIME} for {N} {TIME}s?",
    rf"it (?:receives|gets|takes in|accepts) {N} (?:deliveries|shipments|trucks|truckloads|batches|pallets|loads) "
    rf"of {U}(?: each)?",
    rf"it (?:receives|gets|takes in|accepts) {N} (?:deliveries|shipments|trucks|truckloads|batches|pallets|loads) "
    rf"of (?:unknown|unspecified|an unknown|an unspecified|varying|random) sizes?",
    rf"it (?:ships|sells|sends out|uses|loses) {N} {U} (?:per|a|each|every) {TIME} for "
    rf"(?:several|some|a few|many|a number of|an unknown number of|unknown number of|a couple of) {TIME}s?",
    rf"it (?:ships|sells|sends out|uses|loses) (?:some|several|a few|many|a number of|an unknown number of) {U} "
    rf"(?:per|a|each|every) {TIME} for {N} {TIME}s?",
)]


def _same_unit(word, unit):
    word, unit = word.lower(), unit.lower()
    return word in (unit, unit[:-1], unit[:-2]) and len(word) >= 2


OPS = _ops()


def _unit_groups(rx, m):
    """The unit noun a pattern captured (none for patterns without a unit)."""
    u = m.groupdict().get("u")
    return [u] if u else []


def _sentences(text):
    # split on ". " but keep "(rounded down)." intact
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip().rstrip(".?!").strip() for p in parts if p.strip()]


def solve(task, workspace=None):
    instr = task.get("instruction", "")
    body = _TAIL.sub("", instr).strip()
    sents = _sentences(body)
    if len(sents) < 2:
        return SolveResult.unknown("too few sentences")
    if _VAGUE.search(sents[0]) and not re.search(r"\d", sents[0]):
        return SolveResult.infeasible(f"initial quantity not given: {sents[0]!r}")
    m = _START.match(sents[0])
    if not m:
        return SolveResult.unknown(f"start not recognised: {sents[0]!r}")
    stock = _num(m[2])
    unit = m[3].strip().lower()
    if " " in unit:
        return SolveResult.unknown(f"unit is not a single noun: {unit!r}")
    q = _QUESTION.match(sents[-1])
    if not q:
        return SolveResult.unknown(f"question not recognised: {sents[-1]!r}")
    if q[1].strip().lower() != unit:
        return SolveResult.unknown("question asks about a different unit")
    trace = [stock]
    for sent in sents[1:-1]:
        s = _PREFIX.sub("", sent)
        for rx, fn in OPS:
            mm = rx.match(s)
            if mm and all(_same_unit(g, unit) for g in _unit_groups(rx, mm)):
                new = fn(mm, stock)
                if new is None:
                    return SolveResult.infeasible(f"division by zero in: {sent!r}")
                if new is False:
                    return SolveResult.unknown(f"no rounding rule in: {sent!r}")
                stock = new
                trace.append(stock)
                break
        else:
            if _VAGUE.search(s) and not re.search(r"\d", s):
                return SolveResult.infeasible(f"quantity not given: {sent!r}")
            for rx in MISSING:
                mm = rx.match(s)
                if mm and all(_same_unit(g, unit) for g in _unit_groups(rx, mm)):
                    return SolveResult.infeasible(f"quantity not given: {sent!r}")
            return SolveResult.unknown(f"operation not recognised: {sent!r}")
    return SolveResult.solved(stock, "warehouse template", trace=trace)
