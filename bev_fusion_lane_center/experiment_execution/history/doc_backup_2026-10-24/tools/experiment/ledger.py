"""Ledger of runs: experiment_execution/LEDGER.csv (machine) -> experiment_execution/LEDGER.md (human).

    python ledger.py render                          rewrite LEDGER.md from the csv
    python ledger.py note E209.a --result "S 0.71 night / 0.69 blur" --decision "keep" --notes "seed 0 only"      fill the judgement columns of a row
    python ledger.py add E210.p --hyp "..." --status planned      a row for a run that is only planned (no command yet)

Columns: id, started, ended, status (planned | waiting | running | ok | failed | gpu_busy), exit, duration_s, hypothesis, basis, seed, command, result, decision, notes.
Rule of the loops: a row exists BEFORE the run (planned), is filled by tools/experiment/run.py while it runs, and gets result + decision from the person / agent who reads the numbers.
"""
import argparse
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CSV = os.path.join(ROOT, "experiment_execution", "LEDGER.csv")
MD = os.path.join(ROOT, "experiment_execution", "LEDGER.md")
COLS = ["id", "started", "ended", "status", "exit", "duration_s", "hypothesis", "basis", "seed", "command", "result", "decision", "notes"]


def load():
    if not os.path.exists(CSV):
        return []
    with open(CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows):
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])


def render():
    rows = load()
    lines = ["# Run ledger", "", "Generated from `LEDGER.csv` by `tools/experiment/ledger.py render` - edit the csv (or use `ledger.py note`), not this file.", "",
             "| id | status | hypothesis | basis | seed | result | decision | started | min |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.get("started") or "~"):
        mins = f"{float(r['duration_s'])/60:.0f}" if r.get("duration_s") not in ("", None) else ""
        c = lambda s: (s or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {c(r['id'])} | {c(r['status'])} | {c(r['hypothesis'])} | {c(r['basis'])} | {c(r['seed'])} | {c(r['result'])[:120]} | {c(r['decision'])} | {c(r['started'])[:16]} | {mins} |")
    open(MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(MD, len(rows), "rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "note", "add"])
    ap.add_argument("id", nargs="?")
    for c in ("hyp", "basis", "seed", "result", "decision", "notes", "status"):
        ap.add_argument("--" + c, default=None)
    a = ap.parse_args()
    if a.cmd == "render":
        return render()
    rows = load()
    row = next((r for r in rows if r["id"] == a.id), None)
    if row is None:
        row = dict(id=a.id, status="planned")
        rows.append(row)
    for k, col in (("hyp", "hypothesis"), ("basis", "basis"), ("seed", "seed"), ("result", "result"), ("decision", "decision"), ("notes", "notes"), ("status", "status")):
        if getattr(a, k) is not None:
            row[col] = getattr(a, k)
    save(rows)
    render()


if __name__ == "__main__":
    main()
