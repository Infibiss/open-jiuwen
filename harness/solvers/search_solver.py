"""Deterministic multi-hop lookup over the kb/ markdown knowledge base.

Each kb file looks like:
    # Irina Smith
    Born: 1983
    Employer: Granite AI
    Mentor: Olga Kowalski
The question is turned into a relation chain anchored at one entity name:
    "the headquarters of Sergei Garcia's employer"  -> employer -> headquarters
    "In which year was the mentor of X born?"       -> mentor -> born
Missing entity / attribute / hop target => INFEASIBLE (the KB lacks the data).
"""
import os
import re

from .base import SolveResult

_TAIL = re.compile(r"\s*(Use the file tools|Use bash|Use the|Work only inside|When you are done|Write ONLY)\b.*$", re.I | re.S)
_QWORDS = {"in", "which", "who", "what", "where", "when", "how", "whose", "the", "is", "was", "does", "did", "a", "an", "of", "name"}

# question keyword -> canonical KB attribute (lower-case)
SYNONYMS = [
    (r"born|birth(?:\s*year)?|birthday", "born"),
    (r"employer|employs|employed|works? (?:for|at)|company .{0,20}work", "employer"),
    (r"mentor(?:ed|s)?", "mentor"),
    (r"headquarter(?:s|ed)|hq|based|located", "headquarters"),
    (r"founded|established|founding", "founded"),
]


# Questions that need several KB entries at once (reverse lookups, counts, comparisons):
# not answerable by walking relations from one entity.
_NON_CHAIN = re.compile(
    r"\b(?:mentored|employed|founded|headquartered|managed|owned) by\b|\bhow many\b|\bwho (?:works|worked)\b"
    r"|\bnumber of\b|\bcount\b"
    r"|\b(?:older|younger|oldest|youngest|earlier|earliest|later|latest|newest|first|last|most|least|fewest"
    r"|more|fewer|less|before|after|both|same|all|list)\b"
    # "Who does X mentor?" asks whom X mentors, not X's mentor
    r"|\b(?:does|did|do)\s+.+?\s+(?:mentor|employ|manage|found|own|lead|supervise|teach|train|hire)\b", re.I)


def is_non_chain(question):
    return bool(_NON_CHAIN.search(question))


def load_kb(kb_dir):
    ents = {}
    for root, _, files in os.walk(kb_dir):
        for fn in sorted(files):
            if not fn.endswith((".md", ".txt")):
                continue
            try:
                text = open(os.path.join(root, fn), encoding="utf-8").read()
            except OSError:
                continue
            title, attrs = None, {}
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#") and title is None:
                    title = line.lstrip("#").strip()
                elif ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip() and v.strip():
                        attrs[k.strip().lower()] = v.strip()
            if title is None:
                title = os.path.splitext(fn)[0].split("_", 1)[-1].replace("_", " ").title()
            key = title.lower()
            kind = fn.split("_", 1)[0] if "_" in fn else "|".join(sorted(attrs))
            if key in ents:
                ents[key]["_dup"] = True
            else:
                ents[key] = {"_title": title, "_file": fn, "_kind": kind, **attrs}
    return ents


def _keywords(text, attr_names):
    """Return ordered canonical relations mentioned in text."""
    pats = list(SYNONYMS)
    known = {c for _, c in SYNONYMS}
    for a in attr_names:  # attributes present in this KB but not in SYNONYMS
        if a not in known:
            pats.append((re.escape(a) + r"s?", a))
    found = []
    for pat, canon in pats:
        for m in re.finditer(r"\b(?:" + pat + r")\b", text, re.I):
            found.append((m.start(), canon))
    found.sort()
    out, seen = [], set()
    for p, c in found:
        if p not in seen:
            seen.add(p)
            out.append(c)
    return out


def solve(task, workspace):
    instr = task.get("instruction", "")
    kb_dir = os.path.join(workspace or "", "kb")
    m = re.search(r"The directory (\S+?)/? contains", instr)
    if m:
        kb_dir = os.path.join(workspace or "", m[1])
    if not os.path.isdir(kb_dir):
        return SolveResult.unknown("kb directory not found")
    ents = load_kb(kb_dir)
    if not ents:
        return SolveResult.unknown("empty kb")

    # the actual question: sentence ending with '?'
    body = _TAIL.sub("", instr)
    qs = re.findall(r"[^.?!]*\?", body)
    if not qs:
        return SolveResult.unknown("no question found")
    q = qs[-1].strip()
    # reverse / aggregate / comparative questions are not simple forward chains
    if is_non_chain(q):
        return SolveResult.unknown(f"non-chain question: {q!r}")

    # anchor entity: longest KB title occurring in the question
    hits = [(k, e) for k, e in ents.items() if re.search(r"\b" + re.escape(k) + r"\b", q.lower())]
    hits.sort(key=lambda kv: -len(kv[0]))
    if not hits:
        # A proper name that is not in the KB -> data missing.
        names = re.findall(r"\b([A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)\b", q)
        names = [n for n in names if n.split()[0].lower() not in _QWORDS]
        if names:
            return SolveResult.infeasible(f"entity {names[0]!r} not found in kb")
        return SolveResult.unknown("no entity mentioned")
    key, ent = hits[0]
    # another, non-overlapping entity in the question -> unfamiliar shape
    for k2, _ in hits[1:]:
        if k2 not in key:
            return SolveResult.unknown(f"several entities: {key!r}, {k2!r}")
    if ent.get("_dup"):
        return SolveResult.unknown(f"ambiguous entity {key!r}")

    pos = q.lower().find(key)
    before, after = q[:pos], q[pos + len(key):]
    attr_names = {a for e in ents.values() for a in e if not a.startswith("_")}
    before = re.sub(r"^\s*(?:in|at|for|from)?\s*(?:which|what)\s+\w+", " ", before, flags=re.I)
    poss = re.match(r"^\s*'s?((?:\s+\w+'s?)*\s+\w+)", after)
    poss_rel, rest = [], after
    if poss:
        poss_rel = _keywords(poss[1], attr_names)
        rest = after[poss.end():]
    chain = poss_rel + list(reversed(_keywords(before, attr_names))) + _keywords(rest, attr_names)
    if not chain:
        return SolveResult.unknown(f"no relation recognised in {q!r}")

    cur, path, i = ent, [ent["_title"]], 0
    while i < len(chain):
        rel = chain[i]
        val = cur.get(rel)
        if val is None:
            # only "missing data" if entities of the same kind normally carry this field
            same = [e for e in ents.values() if e.get("_kind") == cur.get("_kind") and e is not cur]
            if same and sum(rel in e for e in same) >= max(1, len(same) // 2):
                return SolveResult.infeasible(f"{cur['_title']} has no '{rel}' in kb", chain=chain)
            return SolveResult.unknown(f"'{rel}' is not a field of {cur['_title']}'s kind", chain=chain)
        path.append(val)
        i += 1
        # same relation mentioned twice ("headquarters ... located") on a
        # non-entity value is one hop, not two
        while i < len(chain) and chain[i] == rel and val.lower() not in ents:
            i += 1
        if i < len(chain):
            nxt = ents.get(val.lower())
            if nxt is None:
                return SolveResult.infeasible(f"no kb entry for {val!r} (needed for '{chain[i]}')", chain=chain)
            if nxt.get("_dup"):
                return SolveResult.unknown(f"ambiguous entity {val!r}")
            cur = nxt
    ans = path[-1]
    ql = q.lower()
    if re.search(r"\bwhich year|what year|\bwhen\b", ql) and not re.fullmatch(r"\d{3,4}", ans):
        return SolveResult.unknown(f"expected a year, got {ans!r}")
    if re.search(r"\bwhich (city|company|person|place|country)|\bwho\b|\bwhere\b", ql) and re.fullmatch(r"\d+", ans):
        return SolveResult.unknown(f"expected a name or place, got {ans!r}")
    return SolveResult.solved(ans, " -> ".join(chain), path=path)
