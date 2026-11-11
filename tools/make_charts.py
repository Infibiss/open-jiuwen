"""README charts as plain SVG (standard library only), one light and one dark file per chart.

    python tools/make_charts.py

Run totals come from the live runs reported in README.md; the forecast chart reads
trajectories/ directly.
"""
import html
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "charts")
FONT = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df", s1="#2a78d6", s2="#eb6834",
                  s3="#1baf7a", on="#ffffff"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#3a3a37", s1="#3987e5", s2="#d95926",
                 s3="#199e70", on="#ffffff"),
}
# (label, points, achievable, tokens)
RUNS = [
    ("Примеры организаторов, 8 задач", 20, 20, 60),
    ("Новые формы задач, seed 1", 63, 63, 2063),
    ("Новые формы задач, seed 2", 51, 51, 2150),
    ("Новые формы задач, seed 3", 52, 52, 1973),
    ("Сложный набор, seed 701", 232, 232, 3229),
    ("Сложный набор, seed 702", 232, 237, 4983),
    ("Сложный набор, seed 703", 230, 230, 4729),
    ("Типовой набор, seed 41", 226, 226, 2134),
    ("Типовой набор, seed 43", 243, 252, 1632),
    ("Типовой набор, seed 44", 231, 234, 2014),
    ("Только LLM, 48 задач, seed 42", 103, 107, 6126),
]
# (label, zero tokens, one LLM call, extra check or escalation), 100 tasks each
CLOSED = [
    ("Типовой набор, seed 41", 82, 17, 1),
    ("Типовой набор, seed 43", 78, 22, 0),
    ("Типовой набор, seed 44", 78, 21, 1),
    ("Сложный набор, seed 701", 60, 39, 1),
    ("Сложный набор, seed 702", 51, 41, 8),
    ("Сложный набор, seed 703", 54, 40, 6),
]
FORECAST_LOGS = ["full_pipeline_run.json", "full_pipeline_hard_seed701.json", "llm_only_run4_fresh_seed42.json"]


def text(x, y, s, fill, size=13, weight=400, anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{html.escape(s)}</text>')


def bar(x, y, w, h, fill, round_end=True, r=4):
    """Horizontal bar anchored at the baseline, rounded only at its data end."""
    r = min(r, w, h / 2) if round_end else 0
    return (f'<path fill="{fill}" d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')


def svg(width, height, t, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}" font-family="{FONT}">'
            f'<rect width="{width}" height="{height}" rx="10" fill="{t["surface"]}"/>' + "".join(body) + "</svg>\n")


def chart_results(t):
    W, row, top, left = 900, 36, 104, 250
    panel, gap = 250, 34
    x2 = left + panel + gap + 62
    H = top + row * len(RUNS) + 18
    low = min(p / a for _, p, a, _ in RUNS) * 100
    top_tok = max(r[3] for r in RUNS)
    title = f"Точность {low:.0f}–100 % при расходе менее {math.ceil(top_tok / 1000):.0f} % бюджета"
    b = [text(24, 36, title, t["text"], 19, 600), text(24, 58, "Живые прогоны на прокси, бюджет 100 000 токенов", t["muted"]),
         text(left, 92, "Баллы, % от доступных", t["muted"], 12), text(x2, 92, "Израсходовано токенов", t["muted"], 12)]
    token_max = 7000
    for i, (name, p, a, tok) in enumerate(RUNS):
        y = top + i * row
        b.append(text(24, y + 16, name, t["text"]))
        b.append(bar(left, y + 2, panel * p / a, 20, t["s1"]))
        b.append(text(left + panel + 8, y + 16, f"{p}/{a}", t["muted"], 12))
        b.append(bar(x2, y + 2, max(3, panel * tok / token_max), 20, t["s1"]))
        b.append(text(x2 + max(3, panel * tok / token_max) + 8, y + 16, f"{tok:,}".replace(",", " "), t["muted"], 12))
    for x in (left, x2):
        b.insert(4, f'<line x1="{x}" x2="{x}" y1="{top - 4}" y2="{H - 16}" stroke="{t["grid"]}"/>')
    return svg(W, H, t, b, title)


def chart_closed(t):
    W, row, top, left = 900, 40, 124, 220
    width = 620
    H = top + row * len(CLOSED) + 16
    lo, hi = min(c[1] for c in CLOSED), max(c[1] for c in CLOSED)
    title = f"{lo}–{hi} из 100 задач закрываются за 0 токенов"
    b = [text(24, 36, title, t["text"], 19, 600), text(24, 58, "Как завершилась каждая задача", t["muted"])]
    x = 24
    for label, col in (("0 токенов", t["s1"]), ("Один вызов LLM", t["s2"]), ("Проверка второй моделью или эскалация", t["s3"])):
        b.append(f'<rect x="{x}" y="80" width="12" height="12" rx="3" fill="{col}"/>')
        b.append(text(x + 18, 91, label, t["muted"], 12))
        x += 18 + 7.2 * len(label) + 26
    for i, (name, zero, one, extra) in enumerate(CLOSED):
        y = top + i * row
        b.append(text(24, y + 18, name, t["text"]))
        cx = left
        parts = [(zero, t["s1"]), (one, t["s2"]), (extra, t["s3"])]
        last = max(j for j, (v, _) in enumerate(parts) if v)
        for j, (v, col) in enumerate(parts):
            if not v:
                continue
            w = width * v / 100
            seg = max(1.0, w - (2 if j != last else 0))
            b.append(bar(cx, y, seg, 26, col, round_end=j == last))
            if w > 34:
                b.append(text(cx + seg / 2, y + 18, str(v), t["on"], 13, 600, "middle"))
            cx += w
        if extra:
            b.append(text(left + width + 8, y + 18, str(extra), t["muted"], 12))
    return svg(W, H, t, b, title)


def forecast_points():
    pts = []
    for name in FORECAST_LOGS:
        with open(os.path.join(ROOT, "trajectories", name), encoding="utf-8") as f:
            for r in json.load(f)["tasks"]:
                p = (r.get("predicted_cost") or {}).get("expected_total")
                if p and r.get("actual_cost"):
                    pts.append((float(p), float(r["actual_cost"])))
    return pts


def chart_forecast(t, pts):
    W, H, L, R, T, B = 640, 560, 76, 28, 104, 62
    lo, hi = 15.0, max(max(a, b) for a, b in pts) * 1.15
    pw, ph = W - L - R, H - T - B
    sx = lambda v: L + pw * math.log(v / lo) / math.log(hi / lo)
    sy = lambda v: T + ph - ph * math.log(v / lo) / math.log(hi / lo)
    mae = st.mean(abs(p - a) for p, a in pts)
    med = st.median(abs(p - a) for p, a in pts)
    title = "Прогноз до запуска близок к факту"
    sub = f"{len(pts)} LLM-задач из трёх журналов: MAE {mae:.0f}, медиана ошибки {med:.0f} токенов"
    b = [text(24, 36, title, t["text"], 19, 600), text(24, 58, sub, t["muted"])]
    for v in (20, 50, 100, 200, 500, 1000):
        if v > hi:
            continue
        b.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{t["grid"]}"/>')
        b.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{T}" y2="{H - B}" stroke="{t["grid"]}"/>')
        b.append(text(L - 8, sy(v) + 4, str(v), t["muted"], 12, anchor="end"))
        b.append(text(sx(v), H - B + 18, str(v), t["muted"], 12, anchor="middle"))
    b.append(f'<line x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}" stroke="{t["muted"]}" '
             f'stroke-width="2" stroke-dasharray="6 5"/>')
    b.append(text(sx(400), sy(400) + 30, "прогноз = факт", t["muted"], 12))
    for p, a in pts:
        b.append(f'<circle cx="{sx(p):.1f}" cy="{sy(a):.1f}" r="4.5" fill="{t["s1"]}" fill-opacity="0.8" '
                 f'stroke="{t["surface"]}" stroke-width="1.5"/>')
    b.append(text(L + pw / 2, H - 16, "Прогноз до запуска, токенов (лог. шкала)", t["muted"], 12, anchor="middle"))
    b.append(f'<g transform="translate(18 {T + ph / 2}) rotate(-90)">'
             + text(0, 0, "Фактический расход, токенов (лог. шкала)", t["muted"], 12, anchor="middle") + "</g>")
    return svg(W, H, t, b, f"{title}. {sub}")


def main():
    os.makedirs(OUT, exist_ok=True)
    pts = forecast_points()
    for mode, t in THEMES.items():
        for name, content in (("results", chart_results(t)), ("closed", chart_closed(t)),
                              ("forecast", chart_forecast(t, pts))):
            with open(os.path.join(OUT, f"{name}_{mode}.svg"), "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
    print(f"wrote 6 charts to {OUT}")


if __name__ == "__main__":
    main()
