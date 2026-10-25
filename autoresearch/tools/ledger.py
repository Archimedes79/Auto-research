"""Ledger of runs: <ledger> (csv, machine) -> <ledger_md> (human); paths from the problem's project.json.

    python ledger.py render                                   rewrite the markdown from the csv
    python ledger.py add E240.a --hyp "..." --pred "..." --rule "..." [--status planned]     a row before the run (planned)
    python ledger.py note E240.a --result "Q 0.741" --decision "KEEP" [--notes "..."]        the judgement columns after the run

Rule: a row exists BEFORE the run (planned), run.py fills it while it runs (status, commit, duration), and the reader of the numbers writes result + decision.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402
from run import COLS  # noqa: E402


def load():
    p = P.path("ledger")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows):
    p = P.path("ledger")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])


def render():
    rows = load()
    lines = ["# Run ledger", "", "Generated from the csv by `autoresearch/tools/ledger.py render` - edit the csv (or use `ledger.py note`), not this file.", "",
             "| id | status | hypothesis | prediction | rule | basis | seed | result | decision | started | min | commit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.get("started") or "~"):
        mins = f"{float(r['duration_s']) / 60:.0f}" if r.get("duration_s") not in ("", None) else ""
        c = lambda s: (s or "").replace("|", "/").replace("\n", " ")  # noqa: E731
        lines.append(f"| {c(r['id'])} | {c(r.get('status'))} | {c(r.get('hypothesis'))} | {c(r.get('prediction'))} | {c(r.get('rule'))} | {c(r.get('basis'))} | {c(r.get('seed'))} | "
                     f"{c(r.get('result'))[:120]} | {c(r.get('decision'))} | {c(r.get('started'))[:16]} | {mins} | {c(r.get('commit'))} |")
    md = P.path("ledger_md")
    open(md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(md, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "note", "add"])
    ap.add_argument("id", nargs="?")
    for c in ("hyp", "basis", "seed", "result", "decision", "notes", "status", "pred", "rule"):
        ap.add_argument("--" + c, default=None)
    a = ap.parse_args()
    if a.cmd == "render":
        return render()
    rows = load()
    row = next((r for r in rows if r["id"] == a.id), None)
    if row is None:
        row = dict(id=a.id, status="planned")
        rows.append(row)
    for k, col in (("hyp", "hypothesis"), ("basis", "basis"), ("seed", "seed"), ("result", "result"), ("decision", "decision"), ("notes", "notes"),
                   ("status", "status"), ("pred", "prediction"), ("rule", "rule")):
        if getattr(a, k) is not None:
            row[col] = getattr(a, k)
    save(rows)
    render()


if __name__ == "__main__":
    main()
