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
W = r"[a-z][a-z\- ]*?"  # unit / container noun (boxes, crates, warehouses, ...)

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


def _num(s):
    s = s.strip().lower().replace(",", "")
    if s.isdigit():
        return int(s)
    return NUM_WORDS[s]


def _ops():
    """(regex, fn(match, stock) -> new stock | None for infeasible)."""
    ops = []

    def add(pattern, fn):
        ops.append((re.compile("^" + pattern + "$", re.I), fn))

    # it ships 36 boxes per day for 3 days
    add(rf"it (?:ships|sells|sends out|uses|loses) {N} {W} (?:per|a|each|every) (?:day|week|month|hour) for {N} (?:days?|weeks?|months?|hours?)",
        lambda m, s: s - _num(m[1]) * _num(m[2]))
    # it receives 2 deliveries of 87 boxes each
    add(rf"it (?:receives|gets|takes in|accepts) {N} (?:deliveries|shipments|trucks|truckloads|batches|pallets|loads) of {N} {W}(?: each)?",
        lambda m, s: s + _num(m[1]) * _num(m[2]))
    # it receives 40 boxes / a delivery of 40 boxes
    add(rf"it (?:receives|gets|buys|takes in|accepts)(?: a (?:delivery|shipment) of)? {N} (?:more |new |extra )?{W}",
        lambda m, s: s + _num(m[1]))
    # it ships 40 boxes
    add(rf"it (?:ships|sells|sends out|gives away|donates|loses|discards|returns|uses) {N} {W}",
        lambda m, s: s - _num(m[1]))
    # 12 boxes are damaged and discarded
    add(rf"{N} {W} (?:are|were|get|got) (?:damaged|lost|stolen|discarded|removed|destroyed|broken)(?: and (?:discarded|removed|thrown away))?",
        lambda m, s: s - _num(m[1]))
    # the stock is split equally into 3 warehouses and it keeps one share (rounded down)
    add(rf"the stock is (?:split|divided) (?:equally|evenly) (?:into|among|between) {N} {W} and it keeps one (?:share|part|portion)(?: \(rounded down\)| rounded down)?",
        lambda m, s: None if _num(m[1]) == 0 else s // _num(m[1]))
    # the stock is doubled (by merging with a sister warehouse)
    add(r"the stock is (doubled|tripled|quadrupled)(?: by .*)?",
        lambda m, s: s * SCALE[m[1].lower()])
    add(rf"the stock is multiplied by {N}(?: by .*)?",
        lambda m, s: s * _num(m[1]))
    add(r"the stock is halved(?: \(rounded down\)| rounded down)?",
        lambda m, s: s // 2)
    add(rf"the stock is divided by {N}(?: \(rounded down\)| rounded down)?",
        lambda m, s: None if _num(m[1]) == 0 else s // _num(m[1]))
    return ops


OPS = _ops()


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
            if mm:
                new = fn(mm, stock)
                if new is None:
                    return SolveResult.infeasible(f"division by zero in: {sent!r}")
                stock = new
                trace.append(stock)
                break
        else:
            if _VAGUE.search(s) and not re.search(r"\d", s):
                return SolveResult.infeasible(f"quantity not given: {sent!r}")
            return SolveResult.unknown(f"operation not recognised: {sent!r}")
    return SolveResult.solved(stock, "warehouse template", trace=trace)
