"""Task shapes the zero-token templates do not know: other math stories, other CSV columns,
a books/authors/publishers knowledge base, other module names, classes, and coding tasks
scored by hidden tests with more checks than the visible file (as the jury does).

    python tools/gen_shape_tasks.py OUT SEED
    python tools/evaluate.py OUT
"""
import json, os, random, sys

out, seed = sys.argv[1], int(sys.argv[2])
R = random.Random(seed)
INF = "__INFEASIBLE__"
TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
CODE_TAIL = (" Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
             "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
tasks = []


def add(tt, instr, exp, files=None, **extra):
    tasks.append((tt, instr, str(exp), files or {}, extra))


# ---------------------------------------------------------------- math (other stories)
a, b, c, d = R.randint(50, 90), R.randint(2, 5), R.randint(40, 70), R.randint(1, 4)
add("math", f"A train travels at {a} km per hour for {b} hours and then at {c} km per hour for {d} hours. How many kilometers does it travel in total? Compute it with the bash tool (python3 -c).", a * b + c * d)
p, n = R.randint(2, 6), R.randint(3, 9)
add("math", f"A shop sells pencils for {p} dollars each. Maria buys {n} pencils and pays with a 100 dollar bill. How much change does she get, in dollars? Compute it with the bash tool (python3 -c).", 100 - p * n)
ben, k, m = R.randint(3, 9), R.randint(2, 4), R.randint(2, 7)
add("math", f"Ben is {ben} years old. Ann is {k} times as old as Ben. Tom is {m} years older than Ann. How old is Tom? Compute it with the bash tool (python3 -c).", ben * k + m)
cap, dr = R.choice([800, 1200, 1600]), R.randint(50, 200)
add("math", f"A tank holds {cap} liters when full. It is three quarters full. Then {dr} liters are drained. How many liters are in the tank now? Compute it with the bash tool (python3 -c).", cap * 3 // 4 - dr)
bk, lo, back, bx, per = R.randint(900, 1500), R.randint(100, 300), R.randint(20, 90), R.randint(2, 4), R.randint(20, 60)
add("math", f"A library has {bk} books. It lends out {lo} books and gets {back} of them back. Then it buys {bx} boxes of {per} books each. How many books are in the library now? Compute it with the bash tool (python3 -c).", bk - lo + back + bx * per)
add("math", "A car travels at a constant speed for 3 hours. How many kilometers does it travel? Compute it with the bash tool (python3 -c).", INF)
add("math", f"A bakery bakes {R.randint(30, 60)} loaves every morning and sells some of them. How many loaves are left at the end of the day? Compute it with the bash tool (python3 -c).", INF)

# ---------------------------------------------------------------- data (other columns)
names = ["Ava", "Liam", "Noah", "Emma", "Mia", "Leo", "Zoe", "Ivan", "Omar", "Lena", "Yuki", "Sara", "Paul", "Nina", "Hugo",
         "Ines", "Raj", "Tara", "Felix", "Iris", "Jon", "Kate", "Max", "Ola", "Pia", "Rui", "Sven", "Tess", "Uma", "Vic",
         "Wes", "Xia", "Yan", "Zak", "Ada", "Bo", "Cy", "Dee", "Eli", "Fay"]
depts = ["engineering", "sales", "support", "finance"]
while True:
    rows = [dict(emp_id=i + 1, name=names[i], department=R.choice(depts), salary=R.randint(40, 160) * 500,
                 year_hired=R.randint(2008, 2023)) for i in range(40)]
    avg = {dp: sum(r["salary"] for r in rows if r["department"] == dp) / max(1, sum(r["department"] == dp for r in rows)) for dp in depts}
    sal = sorted(r["salary"] for r in rows)
    if len(set(avg.values())) == 4 and sal[0] != sal[1] and all(any(r["department"] == dp for r in rows) for dp in depts):
        break
csv = "emp_id,name,department,salary,year_hired\n" + "".join(f"{r['emp_id']},{r['name']},{r['department']},{r['salary']},{r['year_hired']}\n" for r in rows)
H = "The file employees.csv has columns emp_id,name,department,salary,year_hired (40 rows). "
T = " Use bash with python3 to compute it."
F = {"employees.csv": csv}
add("data_analysis", H + "What is the total salary of all employees in the 'engineering' department?" + T, sum(r["salary"] for r in rows if r["department"] == "engineering"), F)
add("data_analysis", H + "How many employees were hired after 2015?" + T, sum(r["year_hired"] > 2015 for r in rows), F)
add("data_analysis", H + "Which department has the highest average salary?" + T, max(avg, key=avg.get), F)
add("data_analysis", H + "What is the name of the employee with the lowest salary?" + T, min(rows, key=lambda r: r["salary"])["name"], F)
add("data_analysis", H + "How many employees in the 'sales' department earn more than 50000?" + T, sum(r["department"] == "sales" and r["salary"] > 50000 for r in rows), F)
add("data_analysis", H + "What is the total bonus paid to the 'sales' department?" + T, INF, F)
add("data_analysis", "The file payroll_2024.csv has columns emp_id,month,amount (480 rows). What was the total amount paid in month 3?" + T, INF, {"README.txt": "payroll export pending\n"})

# ---------------------------------------------------------------- search (other entities)
authors = {"Mira Holt": ("Norway", 1950), "Arlo Vance": ("Canada", 1962), "Selin Kaya": ("Turkey", 1975), "Dario Reyes": ("Chile", 1981)}
pubs = {"Northwind Press": ("Oslo", 1920), "Bluegate Books": ("Toronto", 1955), "Harbor House": ("Lisbon", 1978)}
titles = ["The Silent River", "Glass Orchard", "Winter Ledger", "Salt and Iron", "The Last Lantern", "Paper Moons",
          "Copper Sky", "Dark Harbor", "Quiet Engines"]
books = {}
for t in titles:
    books[t] = (R.choice(list(authors)), R.randint(1985, 2020), R.choice(list(pubs)))
fn = lambda kind, n: f"kb/{kind}_{n.lower().replace(' ', '_')}.md"
files = {fn("book", t): f"# {t}\n\nAuthor: {a_}\nPublished: {y}\nPublisher: {p_}\n" for t, (a_, y, p_) in books.items()}
files.update({fn("author", a_): f"# {a_}\n\nCountry: {c_}\nBorn: {y}\n" for a_, (c_, y) in authors.items()})
files.update({fn("publisher", p_): f"# {p_}\n\nCity: {c_}\nFounded: {y}\n" for p_, (c_, y) in pubs.items()})
HS = "The directory kb/ contains a small knowledge base of markdown files. "
TS = " Use the file tools or bash (ls, grep, cat) to find it."
b0 = "The Silent River"; a0, y0, p0 = books[b0]
add("search", HS + f"In which country was the author of {b0} born?" + TS, authors[a0][0], files)
add("search", HS + f"In which city is the publisher of {b0} based?" + TS, pubs[p0][0], files)
add("search", HS + f"How many books in the knowledge base were written by {a0}?" + TS, sum(v[0] == a0 for v in books.values()), files)
mine = sorted((v[1], t) for t, v in books.items() if v[0] == a0)
if len(mine) > 1 and mine[0][0] != mine[1][0]:
    add("search", HS + f"Which book by {a0} was published first?" + TS, mine[0][1], files)
add("search", HS + f"What literary prize did {b0} win?" + TS, INF, files)
f2 = dict(files)
dh_author = books["Dark Harbor"][0]
del f2[fn("author", dh_author)]
add("search", HS + "In which year was the author of Dark Harbor born?" + TS, INF, f2)
add("search", HS + f"In which year was the publisher of {b0} founded?" + TS, pubs[p0][1], files)

# ---------------------------------------------------------------- coding (other files, classes, hidden tests)
def code_task(mod, src, visible, hidden, exp="tests_pass"):
    test = lambda body: f"import unittest\n\nimport {mod} as m\n\n\nclass T(unittest.TestCase):\n{body}\n\nif __name__ == '__main__':\n    unittest.main()\n"
    instr = (f"Run `python3 test_{mod}.py` in the current directory; some tests fail. Fix the bugs in {mod}.py "
             f"(do not edit the tests) until all tests pass.")
    add("coding", instr + CODE_TAIL, exp, {f"{mod}.py": src, f"test_{mod}.py": test(visible)},
        hidden_tests=test(hidden), module=mod)

code_task("inventory",
          "class Inventory:\n    def __init__(self):\n        self.items = {}\n\n    def add(self, item, qty):\n        self.items[item] = self.items.get(item, 0) + qty\n\n    def remove(self, item, qty):\n        if self.items.get(item, 0) < qty:\n            raise ValueError('not enough')\n        self.items[item] = self.items.get(item, 0) + qty\n\n    def total(self):\n        return sum(self.items.values())\n",
          "    def test_remove(self):\n        inv = m.Inventory()\n        inv.add('a', 5)\n        inv.remove('a', 2)\n        self.assertEqual(inv.items['a'], 3)\n",
          "    def test_remove(self):\n        inv = m.Inventory()\n        inv.add('a', 5)\n        inv.remove('a', 2)\n        self.assertEqual(inv.items['a'], 3)\n\n    def test_more(self):\n        inv = m.Inventory()\n        inv.add('a', 4)\n        inv.add('b', 6)\n        inv.remove('b', 6)\n        self.assertEqual(inv.total(), 4)\n        with self.assertRaises(ValueError):\n            inv.remove('a', 5)\n")
code_task("stats",
          "def mean(xs):\n    return sum(xs) / len(xs)\n\n\ndef median(xs):\n    s = list(xs)\n    n = len(s)\n    if n % 2:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2]) / 2\n",
          "    def test_median(self):\n        self.assertEqual(m.median([3, 1, 2]), 2)\n",
          "    def test_median(self):\n        self.assertEqual(m.median([3, 1, 2]), 2)\n        self.assertEqual(m.median([9, 1, 5, 3]), 4)\n        self.assertEqual(m.median([7]), 7)\n        self.assertEqual(m.mean([1, 2]), 1.5)\n")
code_task("textutils",
          "def capitalize_words(s):\n    return ' '.join(w[:1].lower() + w[1:] for w in s.split())\n\n\ndef is_anagram(a, b):\n    return set(a) == set(b)\n",
          "    def test_cap(self):\n        self.assertEqual(m.capitalize_words('hello world'), 'Hello World')\n\n    def test_anagram(self):\n        self.assertFalse(m.is_anagram('aab', 'abb'))\n",
          "    def test_cap(self):\n        self.assertEqual(m.capitalize_words('hello world'), 'Hello World')\n        self.assertEqual(m.capitalize_words('a b c'), 'A B C')\n\n    def test_anagram(self):\n        self.assertFalse(m.is_anagram('aab', 'abb'))\n        self.assertTrue(m.is_anagram('listen', 'silent'))\n        self.assertFalse(m.is_anagram('ab', 'abc'))\n")
code_task("bank",
          "class Account:\n    def __init__(self, balance=0):\n        self.balance = balance\n\n    def deposit(self, amount):\n        self.balance += amount\n\n    def withdraw(self, amount):\n        if amount < self.balance:\n            raise ValueError('insufficient funds')\n        self.balance -= amount\n",
          "    def test_withdraw(self):\n        acc = m.Account(100)\n        acc.withdraw(30)\n        self.assertEqual(acc.balance, 70)\n",
          "    def test_withdraw(self):\n        acc = m.Account(100)\n        acc.withdraw(30)\n        self.assertEqual(acc.balance, 70)\n        acc.withdraw(70)\n        self.assertEqual(acc.balance, 0)\n        with self.assertRaises(ValueError):\n            acc.withdraw(1)\n")
code_task("shapes",
          "def area_square(s):\n    return s + s\n",
          "    def test_area(self):\n        self.assertEqual(m.area_square(3), 9)\n\n    def test_area_again(self):\n        self.assertEqual(m.area_square(3), 12)\n",
          "    def test_area(self):\n        self.assertEqual(m.area_square(3), 9)\n", exp=INF)

for i, (tt, instr, exp, files_, extra) in enumerate(tasks):
    tid = f"{tt}_{i:04d}"
    d = os.path.join(out, tid)
    os.makedirs(os.path.join(d, "workspace"), exist_ok=True)
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump({"task_id": tid, "task_type": tt, "instruction": instr + ("" if tt == "coding" else TAIL),
                   "reward": R.randint(1, 5), "expected": exp,
                   "check": "unittest" if tt == "coding" else ("exact_int" if exp.lstrip("-").isdigit() else "exact_str"), **extra}, f, indent=2)
    for rel, content in files_.items():
        p_ = os.path.join(d, "workspace", rel)
        os.makedirs(os.path.dirname(p_), exist_ok=True)
        open(p_, "w", encoding="utf-8").write(content)
print("wrote", len(tasks))
