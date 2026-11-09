"""Harder task set: the organisers' templates (most tasks) plus unfamiliar wording, reverse /
count / comparison questions, multi-bug coding tasks and a wide spread of infeasible forms.
Complements gen_tasks.py and checks the LLM ladder on phrasing the zero-token solvers do not know.

    python tools/gen_hard_tasks.py OUT N SEED
    python tools/evaluate.py OUT
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_tasks as G  # noqa: E402

# Coding pool: name -> (correct source, visible assertion, random-input generator, [(old, new) bugs])
R = random.Random(99)
ints = lambda n=8: [R.randint(-20, 40) for _ in range(R.randint(1, n))]
word = lambda: "".join(R.choice("abcxyz") for _ in range(R.randint(1, 6)))
words = lambda: " ".join(word() for _ in range(R.randint(1, 5)))
num = lambda: R.randint(-30, 30)
pos = lambda: R.randint(1, 30)

# name: (correct source, visible assertion, input generator, [(old, new), ...])
POOL = {
 "sum_squares": ("def sum_squares(xs):\n    return sum(x * x for x in xs)\n", "self.assertEqual(m.sum_squares([1, 2, 3]), 14)", lambda: (ints(),), [("x * x", "x + x"), ("sum(", "max(")]),
 "spread": ("def spread(xs):\n    return max(xs) - min(xs)\n", "self.assertEqual(m.spread([1, 5, 3]), 4)", lambda: (ints(),), [("max(xs) - min(xs)", "max(xs) + min(xs)"), ("max(xs) - min(xs)", "min(xs) - max(xs)"), ("min(xs)", "max(xs)")]),
 "area": ("def area(w, h):\n    return w * h\n", "self.assertEqual(m.area(3, 4), 12)", lambda: (pos(), pos()), [("w * h", "w + h"), ("w * h", "w - h"), ("w * h", "w ** h")]),
 "perimeter": ("def perimeter(w, h):\n    return 2 * (w + h)\n", "self.assertEqual(m.perimeter(3, 4), 14)", lambda: (pos(), pos()), [("(w + h)", "(w * h)"), ("2 *", "3 *"), ("(w + h)", "(w - h)")]),
 "smaller": ("def smaller(a, b):\n    return a if a < b else b\n", "self.assertEqual(m.smaller(1, 2), 1)", lambda: (num(), num()), [("a < b", "a > b"), ("else b", "else a"), ("return a if", "return b if")]),
 "weighted": ("def weighted(a, b, c):\n    return a * b + c\n", "self.assertEqual(m.weighted(2, 3, 4), 10)", lambda: (num(), num(), num()), [("a * b + c", "a * b - c"), ("a * b + c", "a + b + c"), ("a * b + c", "a * (b + c)")]),
 "midpoint": ("def midpoint(a, b):\n    return (a + b) / 2\n", "self.assertEqual(m.midpoint(4, 7), 5.5)", lambda: (num(), num()), [("(a + b)", "(a - b)"), ("/ 2", "// 2"), ("/ 2", "/ 3"), ("/ 2", "* 2")]),
 "max_abs": ("def max_abs(xs):\n    return max(abs(x) for x in xs)\n", "self.assertEqual(m.max_abs([-5, 3]), 5)", lambda: (ints(),), [("max(", "min("), ("abs(x)", "x")]),
 "shout": ("def shout(s):\n    return s.upper() + '!'\n", "self.assertEqual(m.shout('hi'), 'HI!')", lambda: (word(),), [("upper", "lower"), ("'!'", "'?'"), ("+ '!'", "* 2")]),
 "has_suffix": ("def has_suffix(s, t):\n    return s.endswith(t)\n", "self.assertTrue(m.has_suffix('hello', 'lo'))", lambda: (word(), R.choice(["a", "x", "yz"])), [("endswith", "startswith")]),
 "has_all": ("def has_all(s, cs):\n    return all(c in s for c in cs)\n", "self.assertFalse(m.has_all('abc', 'ad'))", lambda: (word(), word()[:2]), [("all(", "any("), ("c in s", "c not in s")]),
 "has_any_neg": ("def has_any_neg(xs):\n    return any(x < 0 for x in xs)\n", "self.assertTrue(m.has_any_neg([1, -2, 3]))", lambda: (ints(),), [("any(", "all("), ("x < 0", "x > 0"), ("x < 0", "x < -5")]),
 "count_above": ("def count_above(xs, t):\n    return sum(1 for x in xs if x > t)\n", "self.assertEqual(m.count_above([1, 5, 9], 4), 2)", lambda: (ints(), num()), [("x > t", "x < t"), ("x > t", "x >= t + 1"), ("sum(1", "sum(2")]),
 "count_word": ("def count_word(s, w):\n    return s.split().count(w)\n", "self.assertEqual(m.count_word('a b a', 'a'), 2)", lambda: (" ".join(R.choice("ab") for _ in range(R.randint(1, 6))), R.choice("ab")), [("s.split().count(w)", "s.split().index(w)"), ("s.split()", "s")]),
 "tail_sum": ("def tail_sum(xs):\n    return sum(xs[1:])\n", "self.assertEqual(m.tail_sum([4, 5, 6]), 11)", lambda: (ints(),), [("xs[1:]", "xs[:1]"), ("xs[1:]", "xs[2:]"), ("xs[1:]", "xs[:-1]")]),
 "double_all": ("def double_all(xs):\n    return [x * 2 for x in xs]\n", "self.assertEqual(m.double_all([1, 3]), [2, 6])", lambda: (ints(),), [("x * 2", "x + 2"), ("x * 2", "x ** 2"), ("x * 2", "x * 3")]),
 "evens": ("def evens(xs):\n    return [x for x in xs if x % 2 == 0]\n", "self.assertEqual(m.evens([1, 2, 3, 4]), [2, 4])", lambda: (ints(),), [("== 0", "!= 0"), ("% 2", "% 4"), ("x % 2", "x / 2")]),
 "sorted_desc": ("def sorted_desc(xs):\n    return sorted(xs, reverse=True)\n", "self.assertEqual(m.sorted_desc([1, 3, 2]), [3, 2, 1])", lambda: (ints(),), [(", reverse=True", ""), ("reverse=True", "reverse=False")]),
 "count_loop": ("def count_loop(n):\n    c = 0\n    i = 0\n    while i < n:\n        c += 2\n        i += 1\n    return c\n", "self.assertEqual(m.count_loop(3), 6)", lambda: (R.randint(0, 20),), [("i < n", "i <= n"), ("c += 2", "c += 3"), ("c += 2", "c -= 2")]),
 "gcd": ("def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n", "self.assertEqual(m.gcd(12, 18), 6)", lambda: (pos(), pos()), [("a % b", "a // b"), ("return a", "return b"), ("a, b = b,", "a, b = a,")]),
 "power": ("def power(x, n):\n    return x ** n\n", "self.assertEqual(m.power(2, 3), 8)", lambda: (R.randint(-4, 4), R.randint(0, 5)), [("x ** n", "x * n"), ("x ** n", "n ** x")]),
 "is_even": ("def is_even(x):\n    return x % 2 == 0\n", "self.assertTrue(m.is_even(4))\n        self.assertFalse(m.is_even(3))", lambda: (num(),), [("== 0", "== 1"), ("== 0", "!= 0")]),
 "both_pos": ("def both_pos(a, b):\n    return a > 0 and b > 0\n", "self.assertFalse(m.both_pos(1, -1))", lambda: (num(), num()), [(" and ", " or "), ("b > 0", "b < 0")]),
 "either_neg": ("def either_neg(a, b):\n    return a < 0 or b < 0\n", "self.assertTrue(m.either_neg(1, -1))", lambda: (num(), num()), [(" or ", " and ")]),
 "tax": ("def tax(p, r):\n    return p + p * r\n", "self.assertEqual(m.tax(100, 2), 300)", lambda: (pos(), pos()), [("p + p * r", "p - p * r"), ("p + p * r", "p * p * r"), ("p + p * r", "p + p + r")]),
 "balance": ("def balance(a, b, c):\n    return a - b + c\n", "self.assertEqual(m.balance(10, 3, 2), 9)", lambda: (num(), num(), num()), [("a - b + c", "a - b - c"), ("a - b + c", "a + b + c"), ("a - b + c", "a * b + c")]),
}

INF = "__INFEASIBLE__"
TAIL, CODE_TAIL = G.TAIL, G.CODE_TAIL


def w(p, t):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(t)


def math(r):
    stock = r.randint(500, 2500)
    start = stock
    s = [f"A warehouse starts with {stock} boxes."]
    hard = r.random() < 0.3
    for _ in range(r.randint(2, 5)):
        op = r.choice(["ship", "recv", "split", "double"])
        if op == "ship":
            a, b = r.randint(5, 60), r.randint(1, 5); s.append(f"It ships {a} boxes per day for {b} days."); stock -= a * b
        elif op == "recv":
            a, b = r.randint(1, 4), r.randint(10, 120); s.append(f"It receives {a} deliveries of {b} boxes each."); stock += a * b
        elif op == "split":
            k = r.randint(2, 4); s.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down)."); stock //= k
        else:
            s.append("Then the stock is doubled by merging with a sister warehouse."); stock *= 2
    if hard:
        h = r.choice(["pct", "half", "weeks", "stores", "third"])
        if h == "pct":
            p = r.choice([10, 20, 25, 50]); s.append(f"It sells {p} percent of its stock, rounded down."); stock -= stock * p // 100
        elif h == "half":
            s.append("It ships half of its stock (rounded down)."); stock -= stock // 2
        elif h == "weeks":
            a, b = r.randint(2, 9), r.randint(1, 3); s.append(f"It ships {a} boxes per day for {b} weeks."); stock -= a * b * 7
        elif h == "stores":
            a = r.randint(5, 30); s.append(f"It ships {a} boxes to each of three stores."); stock -= 3 * a
        else:
            s.append("It gives away a third of its stock (rounded down)."); stock -= stock // 3
    inf = r.random() < 0.22
    if inf:
        k = r.choice(["start", "some", "deliv", "nodur", "several", "unspec", "missing_size"])
        i = r.randint(1, len(s))
        if k == "start": s[0] = "A warehouse starts with an unknown number of boxes."
        elif k == "some": s.insert(i, "It ships some boxes per day for 3 days.")
        elif k == "deliv": s.insert(i, "It receives 3 deliveries.")
        elif k == "nodur": s.insert(i, "It ships 36 boxes per day.")
        elif k == "several": s.insert(i, "It ships 20 boxes per day for several days.")
        elif k == "unspec": s.insert(i, "Then the stock is split equally into an unspecified number of warehouses and it keeps one share (rounded down).")
        else: s.insert(i, "It receives 2 deliveries of unknown size.")
    t = " ".join(s) + " How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    return t + TAIL, (INF if inf else str(stock)), "exact_int", {}, {}


def data(r):
    n = r.choice([20, 50, 120, 300])
    P = ["widget", "gadget", "gizmo", "sprocket"]; RG = ["north", "south", "east", "west"]
    rows = [dict(order_id=i, region=r.choice(RG), product=r.choice(P), month=r.randint(1, 12), units=r.randint(1, 40), price=r.randint(5, 99)) for i in range(1, n + 1)]
    csvt = "order_id,region,product,month,units,price\n" + "".join(f"{x['order_id']},{x['region']},{x['product']},{x['month']},{x['units']},{x['price']}\n" for x in rows)
    files = {"sales.csv": csvt}
    p = r.choice(sorted({x["product"] for x in rows})); a = r.randint(1, 10); b = r.randint(a, min(12, a + 3)); g = r.choice(RG)
    S = lambda f: sum(x["units"] for x in rows if f(x))
    pm = lambda x: x["product"] == p and a <= x["month"] <= b
    inf = r.random() < 0.22
    if inf:
        k = r.choice(["product", "column", "month", "year", "file", "region"])
        if k == "product": q = f"How many units of product 'doohickey' were sold in months {a} through {b} inclusive?"
        elif k == "column": q = f"What is the total discount given on product '{p}' in months {a} through {b} inclusive?"
        elif k == "month": q = f"How many units of product '{p}' were sold in months 13 through 14 inclusive?"
        elif k == "year": q = f"How many units of product '{p}' were sold in the year 2021?"
        elif k == "region": q = f"How many units of product '{p}' were sold in region 'atlantis'?"
        else:
            del files["sales.csv"]; files["notes.txt"] = "sales data will be uploaded later\n"
            q = f"How many units of product '{p}' were sold in months {a} through {b} inclusive?"
        exp = INF
    else:
        k = r.random()
        if k < .55: q, exp = f"How many units of product '{p}' were sold in months {a} through {b} inclusive?", S(pm)
        elif k < .65: q, exp = f"How many units of product '{p}' were sold in region '{g}' in months {a} through {b} inclusive?", S(lambda x: pm(x) and x["region"] == g)
        elif k < .75:
            t = r.randint(20, 70); q, exp = f"How many units of product '{p}' were sold at a price above {t}?", S(lambda x: x["product"] == p and x["price"] > t)
        elif k < .85:
            t = r.randint(10, 30); q, exp = f"How many orders of product '{p}' had more than {t} units?", sum(1 for x in rows if x["product"] == p and x["units"] > t)
        elif k < .93:
            g2 = r.choice([x for x in RG if x != g]); q, exp = f"How many units of product '{p}' were sold in region '{g}' or region '{g2}'?", S(lambda x: x["product"] == p and x["region"] in (g, g2))
        else: q, exp = f"How many units of product '{p}' were sold outside region '{g}'?", S(lambda x: x["product"] == p and x["region"] != g)
    t = f"The file sales.csv has columns order_id,region,product,month,units,price ({n} rows). {q} Use bash with python3 to compute it."
    return t + TAIL, str(exp), "exact_int", files, {}


def search(r):
    ppl = r.sample([f"{f} {l}" for f in G.FIRST for l in G.LAST], 30)
    comp = {c: dict(hq=r.choice(G.CITIES), founded=r.randint(1950, 2020)) for c in G.COMPANIES}
    pers = {p: dict(born=r.randint(1955, 2003), emp=r.choice(G.COMPANIES), men=r.choice([x for x in ppl if x != p])) for p in ppl}
    fname = lambda kind, n: f"kb/{kind}_{n.lower().replace(' ', '_')}.md"
    files = {fname("company", c): f"# {c}\n\nHeadquarters: {d['hq']}\nFounded: {d['founded']}\n" for c, d in comp.items()}
    for p, d in pers.items():
        files[fname("person", p)] = f"# {p}\n\nBorn: {d['born']}\nEmployer: {d['emp']}\nMentor: {d['men']}\n"
    p = r.choice(ppl); d = pers[p]
    inf = r.random() < 0.22
    if inf:
        k = r.choice(["person", "field", "dangling", "attr", "ceo", "mentor_gone"])
        if k == "person":
            x = r.choice([f"{f} {l}" for f in G.FIRST for l in G.LAST if f"{f} {l}" not in ppl]); q = f"In which year was {x} born?"
        elif k == "field":
            files[fname("person", p)] = f"# {p}\n\nEmployer: {d['emp']}\nMentor: {d['men']}\n"; q = f"In which year was {p} born?"
        elif k == "dangling":
            del files[fname("company", d["emp"])]; q = f"In which city is the headquarters of {p}'s employer?"
        elif k == "attr": q = f"What is the salary of {p}?"
        elif k == "ceo": q = f"Who is the CEO of {d['emp']}?"
        else:
            del files[fname("person", d["men"])]; q = f"In which year was {p}'s mentor born?"
        exp = INF
    else:
        k = r.randrange(8)
        if k == 0: q, exp = f"In which year was {p} born?", d["born"]
        elif k == 1: q, exp = f"In which city is the headquarters of {p}'s employer?", comp[d["emp"]]["hq"]
        elif k == 2: q, exp = f"In which year was {p}'s mentor's employer founded?", comp[pers[d["men"]]["emp"]]["founded"]
        elif k == 3: q, exp = f"Who is the mentor of {p}'s mentor?", pers[d["men"]]["men"]
        elif k == 4: q, exp = f"Which city hosts the head office of the company where {p} works?", comp[d["emp"]]["hq"]
        elif k == 5:
            c = d["emp"]; q, exp = f"How many people in the knowledge base work at {c}?", sum(1 for v in pers.values() if v["emp"] == c)
        elif k == 6:
            a2 = r.choice([x for x in ppl if x != p and pers[x]["born"] != d["born"]]); q, exp = f"Who is older, {p} or {a2}?", (p if d["born"] < pers[a2]["born"] else a2)
        else: q, exp = f"Who was born first, {p} or {d['men']}?", (p if d["born"] < pers[d["men"]]["born"] else d["men"]) if d["born"] != pers[d["men"]]["born"] else p
    t = f"The directory kb/ contains a small knowledge base of markdown files. {q} Use the file tools or bash (ls, grep, cat) to find it."
    return t + TAIL, str(exp), "exact_str", files, {}


def coding(r):
    src_pool = {k: (v[0], v[1], v[3]) for k, v in POOL.items()}
    for k, (g, a) in G.CORRECT.items():
        src_pool.setdefault(k, (g, a, [b for b in G.BUGS if b[0] in g]))
    names = r.sample(sorted(src_pool), r.randint(4, 6))
    lib, ref, tests, bugged = [], [], [], []
    want = r.randint(1, 3)
    for nm in names:
        g, a, bugs = src_pool[nm]; bugs = list(bugs); r.shuffle(bugs); s = g; ref.append(g)
        if len(bugged) < want:
            for x, y in bugs:
                c = g.replace(x, y, 1)
                if c != g and G._bug_visible(c, nm, a):
                    s = c; bugged.append(nm); break
        lib.append(s); tests.append(f"    def test_{nm}(self):\n        {a}\n")
    inf = r.random() < 0.2 or not bugged
    if inf:
        nm = names[0]; a = src_pool[nm][1].splitlines()[0].strip()
        k = r.choice(["neq", "two"])
        if k == "neq" or "assertEqual" not in a:
            a2 = a.replace("assertEqual(", "assertNotEqual(") if "assertEqual" in a else (a.replace("assertTrue", "assertFalse") if "assertTrue" in a else a.replace("assertFalse", "assertTrue"))
        else:
            head, tail = a.rsplit(",", 1); a2 = head + ", 987654)" if ")" in tail else a
        tests.append(f"    def test_{nm}_again(self):\n        {a2}\n")
    test = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n" + "\n".join(tests) + "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    t = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not edit the tests) until all tests pass.")
    return t + CODE_TAIL, (INF if inf else "tests_pass"), "unittest", {"mylib.py": "\n\n".join(lib), "test_mylib.py": test}, {"reference_solution": "\n\n".join(ref), "bugged_functions": bugged}


GEN = {"math": math, "data_analysis": data, "search": search, "coding": coding}
out, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = random.Random(seed)
for i in range(n):
    tt = list(GEN)[i % 4]
    ins, exp, chk, files, extra = GEN[tt](r)
    tid = f"{tt}_{i:04d}"; d = os.path.join(out, tid)
    w(os.path.join(d, "task.json"), json.dumps({"task_id": tid, "task_type": tt, "instruction": ins, "reward": r.randint(1, 5), "expected": exp, "check": chk, **extra}, indent=2))
    os.makedirs(os.path.join(d, "workspace"), exist_ok=True)
    for rel, c in files.items():
        w(os.path.join(d, "workspace", rel), c)
print("wrote", n)
