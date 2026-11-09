"""Synthetic training-set generator mirroring the organisers' task format.

    python tools/gen_tasks.py --out /tmp/train --n 100 --seed 1

Produces <out>/<type>_<NNNN>/task.json (+ workspace/) with the hidden
`expected` field. Infeasible tasks (~20%) get expected = "__INFEASIBLE__".
Beside the four sample templates it emits paraphrases / extra operations so
the harness's escalation path (template not recognised -> LLM) gets exercised.
"""
import argparse
import json
import os
import random

INF = "__INFEASIBLE__"
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODE_TAIL = (" Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
             "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")

FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya", "Nikolai",
         "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Novak", "Smith", "Haddad", "Wang", "Kowalski", "Petrov", "Sato", "Garcia", "Muller", "Ivanova"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Medan", "Tbilisi", "Hanoi", "Osaka", "Quito", "Izmir", "Lisbon", "Accra", "Leeds", "Graz", "Tallinn"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]
REGIONS = ["north", "south", "east", "west"]


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------------------------------------------- math
def gen_math(rng):
    stock = rng.randint(500, 2000)
    sents = [f"A warehouse starts with {stock} boxes."]
    infeasible = rng.random() < 0.2
    variant = rng.random()
    for _ in range(rng.randint(2, 5)):
        op = rng.choice(["ship", "recv", "split", "double", "lose", "recv1", "triple"] if variant < 0.3
                        else ["ship", "recv", "split", "double"])
        if op == "ship":
            a, b = rng.randint(5, 60), rng.randint(1, 5)
            sents.append(f"It ships {a} boxes per day for {b} days.")
            stock -= a * b
        elif op == "recv":
            a, b = rng.randint(1, 4), rng.randint(10, 120)
            sents.append(f"It receives {a} deliveries of {b} boxes each.")
            stock += a * b
        elif op == "split":
            k = rng.randint(2, 4)
            sents.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down).")
            stock //= k
        elif op == "double":
            sents.append("Then the stock is doubled by merging with a sister warehouse.")
            stock *= 2
        elif op == "triple":
            sents.append("Then the stock is tripled.")
            stock *= 3
        elif op == "lose":
            a = rng.randint(3, 40)
            sents.append(f"Then {a} boxes are damaged and discarded.")
            stock -= a
        elif op == "recv1":
            a = rng.randint(10, 90)
            sents.append(f"It receives a delivery of {a} boxes.")
            stock += a
    if infeasible:
        kind = rng.choice(["some", "unknown_start", "zero_split"])
        if kind == "some":
            sents.insert(rng.randint(1, len(sents)), "It ships some boxes per day for a few days.")
        elif kind == "unknown_start":
            sents[0] = "A warehouse starts with an unknown number of boxes."
        else:
            sents.insert(rng.randint(1, len(sents)),
                         "Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).")
    text = " ".join(sents) + " How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    return text + TAIL, (INF if infeasible else str(stock)), "exact_int", {}


# ----------------------------------------------------------------- data analysis
def gen_data(rng):
    n = rng.choice([20, 50, 120, 300])
    rows = []
    for i in range(1, n + 1):
        rows.append({"order_id": i, "region": rng.choice(REGIONS), "product": rng.choice(PRODUCTS[:3] if n == 20 else PRODUCTS),
                     "month": rng.randint(1, 12), "units": rng.randint(1, 40), "price": rng.randint(5, 99)})
    files = {}
    buf = ["order_id,region,product,month,units,price"]
    buf += [f"{r['order_id']},{r['region']},{r['product']},{r['month']},{r['units']},{r['price']}" for r in rows]
    files["sales.csv"] = "\n".join(buf) + "\n"
    present = sorted({r["product"] for r in rows})
    p = rng.choice(present)
    a = rng.randint(1, 11)
    b = rng.randint(a, min(12, a + rng.randint(0, 3)))
    kind = rng.random()
    infeasible = rng.random() < 0.2
    if infeasible:
        ik = rng.choice(["absent_product", "bad_months", "missing_col"])
        if ik == "absent_product":
            p = "doohickey"
            q = f"How many units of product '{p}' were sold in months {a} through {b} inclusive?"
        elif ik == "bad_months":
            q = f"How many units of product '{p}' were sold in months 13 through 14 inclusive?"
        else:
            q = f"What is the total discount given on product '{p}' in months {a} through {b} inclusive?"
        exp = INF
    elif kind < 0.6:
        q = f"How many units of product '{p}' were sold in months {a} through {b} inclusive?"
        exp = sum(r["units"] for r in rows if r["product"] == p and a <= r["month"] <= b)
    elif kind < 0.7:
        reg = rng.choice(REGIONS)
        q = f"What is the total revenue (units * price) of product '{p}' in region '{reg}'?"
        exp = sum(r["units"] * r["price"] for r in rows if r["product"] == p and r["region"] == reg)
    elif kind < 0.8:
        q = f"How many orders of product '{p}' were placed in months {a} through {b} inclusive?"
        exp = sum(1 for r in rows if r["product"] == p and a <= r["month"] <= b)
    elif kind < 0.9:
        q = f"Which region sold the most units of product '{p}'? Answer with the region name."
        tot = {}
        for r in rows:
            if r["product"] == p:
                tot[r["region"]] = tot.get(r["region"], 0) + r["units"]
        best = sorted(tot.items(), key=lambda kv: -kv[1])
        if len(best) > 1 and best[0][1] == best[1][1]:
            return gen_data(rng)
        exp = best[0][0]
    else:
        q = f"What is the largest number of units in a single order of product '{p}' in month {a}?"
        sel = [r["units"] for r in rows if r["product"] == p and r["month"] == a]
        if not sel:
            return gen_data(rng)
        exp = max(sel)
    text = (f"The file sales.csv has columns order_id,region,product,month,units,price ({n} rows). {q} "
            f"Use bash with python3 to compute it.")
    return text + TAIL, str(exp), "exact_int" if str(exp).isdigit() else "exact_str", files


# ----------------------------------------------------------------- search
def gen_search(rng):
    people = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 28)
    comp = {c: {"hq": rng.choice(CITIES), "founded": rng.randint(1950, 2020)} for c in COMPANIES}
    pers = {}
    for p in people:
        pers[p] = {"born": rng.randint(1955, 2003), "employer": rng.choice(COMPANIES),
                   "mentor": rng.choice([x for x in people if x != p])}
    files = {}
    for c, d in comp.items():
        files[f"kb/company_{c.lower().replace(' ', '_')}.md"] = f"# {c}\n\nHeadquarters: {d['hq']}\nFounded: {d['founded']}\n"
    for p, d in pers.items():
        files[f"kb/person_{p.lower().replace(' ', '_')}.md"] = \
            f"# {p}\n\nBorn: {d['born']}\nEmployer: {d['employer']}\nMentor: {d['mentor']}\n"
    p = rng.choice(people)
    kind = rng.randrange(6)
    infeasible = rng.random() < 0.2
    if infeasible:
        ik = rng.choice(["absent_person", "absent_company"])
        if ik == "absent_person":
            p = rng.choice([f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" not in people])
            q, exp = f"In which year was {p} born?", INF
        else:
            emp = pers[p]["employer"]
            del files[f"kb/company_{emp.lower().replace(' ', '_')}.md"]
            q, exp = f"In which city is the headquarters of {p}'s employer?", INF
    elif kind == 0:
        q, exp = f"In which year was {p} born?", pers[p]["born"]
    elif kind == 1:
        q, exp = f"In which city is the headquarters of {p}'s employer?", comp[pers[p]["employer"]]["hq"]
    elif kind == 2:
        q, exp = f"Who is the mentor of {p}?", pers[p]["mentor"]
    elif kind == 3:
        q, exp = f"In which year was {p}'s mentor born?", pers[pers[p]["mentor"]]["born"]
    elif kind == 4:
        q, exp = f"In which year was the employer of {p}'s mentor founded?", comp[pers[pers[p]["mentor"]]["employer"]]["founded"]
    else:
        q, exp = f"Which company does {p} work for?", pers[p]["employer"]
    text = (f"The directory kb/ contains a small knowledge base of markdown files. {q} "
            f"Use the file tools or bash (ls, grep, cat) to find it.")
    return text + TAIL, str(exp), "exact_str", files


# ----------------------------------------------------------------- coding
CORRECT = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      "self.assertEqual(m.reverse_words('c a b'), 'b a c')"),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              "self.assertEqual(m.total([1, 2, 3]), 6)"),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                "self.assertEqual(m.maximum([3, 9, 2]), 9)"),
    "minimum": ("def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                "self.assertEqual(m.minimum([3, 1, 2]), 1)"),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n", "self.assertEqual(m.average([2, 4, 6]), 4)"),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n", "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"),
    "is_palindrome": ("def is_palindrome(s):\n    return s == s[::-1]\n",
                      "self.assertTrue(m.is_palindrome('abba'))\n        self.assertFalse(m.is_palindrome('abc'))"),
    "factorial": ("def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
                  "self.assertEqual(m.factorial(5), 120)"),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
                     "self.assertEqual(m.count_vowels('Hello World'), 3)"),
    "square_all": ("def square_all(xs):\n    return [x * x for x in xs]\n",
                   "self.assertEqual(m.square_all([1, 2, 3]), [1, 4, 9])"),
    "last_n": ("def last_n(xs, n):\n    return xs[-n:]\n", "self.assertEqual(m.last_n([5, 6, 7, 8], 2), [7, 8])"),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              "self.assertEqual(m.clamp(15, 0, 10), 10)\n        self.assertEqual(m.clamp(-5, 0, 10), 0)"),
    "fib": ("def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
            "self.assertEqual(m.fib(10), 55)"),
    "word_count": ("def word_count(s):\n    return len(s.split())\n", "self.assertEqual(m.word_count('a b  c'), 3)"),
    "product": ("def product(xs):\n    p = 1\n    for x in xs:\n        p *= x\n    return p\n",
                "self.assertEqual(m.product([2, 3, 4]), 24)"),
    "first": ("def first(xs):\n    return xs[0]\n", "self.assertEqual(m.first([4, 5, 6]), 4)"),
    "last": ("def last(xs):\n    return xs[-1]\n", "self.assertEqual(m.last([4, 5, 6]), 6)"),
    "contains": ("def contains(xs, v):\n    return v in xs\n", "self.assertTrue(m.contains([1, 2, 3], 2))"),
    "is_sorted": ("def is_sorted(xs):\n    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))\n",
                  "self.assertFalse(m.is_sorted([1, 3, 2]))"),
    "abs_diff": ("def abs_diff(a, b):\n    return abs(a - b)\n", "self.assertEqual(m.abs_diff(3, 7), 4)"),
    "count_positive": ("def count_positive(xs):\n    return sum(1 for x in xs if x > 0)\n",
                       "self.assertEqual(m.count_positive([-1, 2, 0, 5]), 2)"),
}
BUGS = [("s += x", "s -= x"), ("x > best", "x < best"), ("x < best", "x > best"), ("reversed(", "sorted("),
        ("% 2 == 0", "% 2 == 1"), ("xs[:n]", "xs[:n - 1]"), ("s == s[::-1]", "s == s[::1]"), ("n + 1", "n"),
        ("r *= i", "r += i"), ("in 'aeiou'", "not in 'aeiou'"), ("x * x", "x + x"), ("xs[-n:]", "xs[n:]"),
        ("max(lo, min(x, hi))", "min(lo, max(x, hi))"), ("a + b", "a * b"), ("/ len(xs)", "// len(xs) + 1"),
        ("p *= x", "p += x"), ("len(s.split())", "len(s.split(' '))"),
        # "other plausible one-token bugs" (reviewer's set): constants, indices, slices, operators
        ("/ len(xs)", "* len(xs)"), ("for x in xs:", "for x in xs[1:]:"), ("xs[0]", "xs[1]"), ("xs[-1]", "xs[0]"),
        ("p = 1", "p = 0"), ("range(2, n + 1)", "range(2, n)"), ("range(n)", "range(n - 1)"), ("a + b", "a - b"),
        ("% 2 == 0", "% 3 == 0"), ("x * x", "x ** 3"), ("x * x", "x * 2"), ("v in xs", "v not in xs"),
        ("<= xs[i + 1]", ">= xs[i + 1]"), ("abs(a - b)", "abs(a + b)"), ("x > 0", "x >= 0"), ("x > 0", "x < 0"),
        ("c in 'aeiou'", "c in 'aeiouy'"), ("s.lower()", "s.upper()")]


def _bug_visible(buggy_src, fname, assertion):
    """True if the single visible assertion fails on the buggy function."""
    import types
    import unittest
    ns = {}
    try:
        exec(buggy_src, ns)
        tc = unittest.TestCase()
        exec(assertion.replace("self.", "tc.").replace("\n        ", "\n"),
             {"tc": tc, "m": types.SimpleNamespace(**{fname: ns[fname]})})
    except Exception:  # noqa: BLE001 - assertion failed or the bug crashes: visible
        return True
    return False


def gen_coding(rng):
    names = rng.sample(sorted(CORRECT), rng.randint(4, 6))
    lib, ref, tests, bugged = [], [], [], []
    for nm in names:
        src, assertion = CORRECT[nm]
        ref.append(src)
        bugs = [b for b in BUGS if b[0] in src]
        rng.shuffle(bugs)
        if bugs and (not bugged or rng.random() < 0.5):
            for a, b in bugs:  # only bugs the visible test can see (others no harness can find)
                cand = src.replace(a, b, 1)
                if _bug_visible(cand, nm, assertion):
                    src = cand
                    bugged.append(nm)
                    break
        lib.append(src)
        tests.append(f"    def test_{nm}(self):\n        {assertion}\n")
    infeasible = rng.random() < 0.15 or not bugged
    if infeasible:  # contradictory requirement on the same call
        nm = names[0]
        tests.append(f"    def test_{nm}_again(self):\n        "
                     + CORRECT[nm][1].replace("assertEqual(", "assertNotEqual(").replace("assertTrue(", "assertFalse(", 1)
                     .replace("assertFalse(m.is_sorted", "assertTrue(m.is_sorted") + "\n")
    test = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n" + "\n".join(tests) + \
           "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    files = {"mylib.py": "\n\n".join(lib), "test_mylib.py": test}
    text = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
            "(do not edit the tests) until all tests pass.")
    extra = {"reference_solution": "\n\n".join(ref), "bugged_functions": bugged}
    return text + CODE_TAIL, (INF if infeasible else "tests_pass"), "unittest", files, extra


GENS = {"math": gen_math, "data_analysis": gen_data, "search": gen_search, "coding": gen_coding}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    types = list(GENS)
    for i in range(args.n):
        tt = types[i % 4]
        res = GENS[tt](rng)
        instr, exp, check, files = res[:4]
        extra = res[4] if len(res) > 4 else {}
        tid = f"{tt}_{i:04d}"
        d = os.path.join(args.out, tid)
        task = {"task_id": tid, "task_type": tt, "instruction": instr, "reward": rng.randint(1, 5),
                "expected": exp, "check": check, **extra}
        w(os.path.join(d, "task.json"), json.dumps(task, indent=2))
        if files or tt != "math":
            os.makedirs(os.path.join(d, "workspace"), exist_ok=True)
        for rel, content in files.items():
            w(os.path.join(d, "workspace", rel), content)
    print(f"wrote {args.n} tasks to {args.out}")


if __name__ == "__main__":
    main()
