"""Ledger of runs: <ledger> (csv, machine) -> <ledger_md> (human); paths from the problem's project.json.

    python ledger.py render                                   rewrite the markdown from the csv
    python ledger.py add E240.a --hyp "..." --pred "..." --rule "..." [--status planned]     a row before the run (planned)
    python ledger.py note E240.a --result "Q 0.741" --decision "KEEP" [--notes "..."]        the judgement columns after the run
    python ledger.py tail [--loop 19] [-n 15]                 the last rows (of one loop) in short form, for a wake-up instead of the whole LEDGER.md

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
    """whole-file write under the ledger lock (take P.locked yourself when you load, change and save: see main)"""
    import io
    p = P.path("ledger")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])
    P.atomic_write(p, buf.getvalue())


def render():
    rows = load()
    lines = ["# Run ledger", "", "Generated from the csv by `autoresearch/tools/ledger.py render` - edit the csv (or use `ledger.py note`), not this file.", ""]
    import metrics  # the loop table (ROADMAP P3) at the head: how much each loop ran, decided and predicted
    lines += ["## Loops", "", "From `autoresearch/tools/metrics.py` (columns explained there; `-v` shows how each row was counted).", "",
              metrics.table(metrics.collect(), md=True), "", "## Runs", ""]
    lines += ["| id | status | hypothesis | prediction | rule | basis | seed | result | decision | started | min | commit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.get("started") or "~"):
        mins = f"{float(r['duration_s']) / 60:.0f}" if r.get("duration_s") not in ("", None) else ""
        c = lambda s: (s or "").replace("|", "/").replace("\n", " ")  # noqa: E731
        lines.append(f"| {c(r['id'])} | {c(r.get('status'))} | {c(r.get('hypothesis'))} | {c(r.get('prediction'))} | {c(r.get('rule'))} | {c(r.get('basis'))} | {c(r.get('seed'))} | "
                     f"{c(r.get('result'))[:120]} | {c(r.get('decision'))} | {c(r.get('started'))[:16]} | {mins} | {c(r.get('commit'))} |")
    md = P.path("ledger_md")
    open(md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(md, len(rows), "rows")


def tail(loop=None, n=15):
    import metrics
    rows = load()
    if loop:
        lo, hi = P.cfg().get("loops", {})[loop]
        rows = [r for r in rows if (k := metrics.number(r["id"])) is not None and lo <= k and (hi is None or k <= hi)]
    rows = sorted(rows, key=lambda r: r.get("started") or "~")[-n:]
    for r in rows:
        print(f"{r['id']:12} {r.get('status', ''):8} {(r.get('started') or '')[5:16]:11} | {(r.get('result') or '')[:70]} | {(r.get('decision') or '')[:60]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "note", "add", "tail"])
    ap.add_argument("id", nargs="?")
    for c in ("hyp", "basis", "seed", "result", "decision", "notes", "status", "pred", "rule", "loop"):
        ap.add_argument("--" + c, default=None)
    ap.add_argument("-n", type=int, default=15)
    a = ap.parse_args()
    if a.cmd == "render":
        return render()
    if a.cmd == "tail":
        return tail(a.loop, a.n)
    with P.locked(P.path("ledger")):  # read, change and write as one step: two chats noting at once lose nothing
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
