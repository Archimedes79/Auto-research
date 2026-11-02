"""Ledger of runs: <ledger> (csv, machine) -> <ledger_md> (human); paths from the problem's project.json.

    python ledger.py render                                   rewrite the markdown from the csv
    python ledger.py add E240.a --hyp "..." --pred "..." --rule "..." [--status planned]     a row before the run (planned)
    python ledger.py note E240.a --result "Q 0.741" --decision "KEEP" [--notes "..."]        the judgement columns after the run
    python ledger.py tail [--loop 19] [-n 15]                 the last rows (of one loop) in short form, then the runs in progress, for a wake-up
    python ledger.py sync [--dry-run]                         add the row of every ended run that has none, also a run whose launcher died (stalled); facts from its run record.
                                                              Also replaces the finished row of an EARLIER attempt of an id by the record of a later one (a re-run whose own
                                                              final write failed); the earlier result and decision are kept as a note of the new row

Rule: a card row exists BEFORE the run (planned, optional), run.py writes the facts when the run ENDS (status, commit, duration), and the reader of the numbers
writes result + decision. A run in progress is not a ledger row: it is read from the run records (`tail`, schedule.py show), so there is one place for it.
"""
import argparse
import csv
import glob
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
    try:  # the loop table (ROADMAP P3) at the head: how much each loop ran, decided and predicted; it never stops a render
        import metrics
        loops = metrics.table(metrics.collect(), md=True)
    except Exception as e:  # noqa: BLE001
        loops = f"(metrics.py failed: {e!r})"
    lines += ["## Loops", "", "From `autoresearch/tools/metrics.py` (columns explained there; `-v` shows how each row was counted).", "",
              loops, "", "## Runs", ""]
    lines += ["| id | status | hypothesis | prediction | rule | basis | seed | result | decision | started | min | commit |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r.get("started") or "~"):
        mins = f"{float(r['duration_s']) / 60:.0f}" if r.get("duration_s") not in ("", None) else ""
        c = lambda s: (s or "").replace("|", "/").replace("\n", " ")  # noqa: E731
        lines.append(f"| {c(r['id'])} | {c(r.get('status'))} | {c(r.get('hypothesis'))} | {c(r.get('prediction'))} | {c(r.get('rule'))} | {c(r.get('basis'))} | {c(r.get('seed'))} | "
                     f"{c(r.get('result'))[:120]} | {c(r.get('decision'))} | {c(r.get('started'))[:16]} | {mins} | {c(r.get('commit'))} |")
    md = P.path("ledger_md")
    P.atomic_write(md, "\n".join(lines) + "\n")
    print(md, len(rows), "rows")


def run_dirs():
    """(id, run directory) of every run record"""
    records = P.path("run_records")
    return [(os.path.basename(d), d) for d in sorted(glob.glob(os.path.join(records, "*"))) if os.path.isfile(os.path.join(d, "run.json"))]


def tail(loop=None, n=15):
    import metrics
    import run as RUN
    rows = load()
    inside = lambda rid: True  # noqa: E731
    if loop:
        loops = P.cfg().get("loops", {})
        if loop not in loops:
            sys.exit(f"loop {loop} is not in project.json 'loops' ({', '.join(loops) or 'none'})")
        lo, hi = loops[loop]
        inside = lambda rid: (k := metrics.number(rid)) is not None and lo <= k and (hi is None or k <= hi)  # noqa: E731
    rows = sorted([r for r in rows if inside(r["id"])], key=lambda r: r.get("started") or "~")[-n:]
    for r in rows:
        print(f"{r['id']:12} {r.get('status', ''):8} {(r.get('started') or '')[5:16]:11} | {(r.get('result') or '')[:70]} | {(r.get('decision') or '')[:60]}")
    lost = [rid for rid, d, st in missing(inside)]
    if lost:
        print(f"{len(lost)} run(s) ended or lost without a current ledger row ({', '.join(lost[:6])}{' ...' if len(lost) > 6 else ''}): run `ledger.py sync`")
    live = [(rid, d) for rid, d in run_dirs() if inside(rid) and RUN.state_of(d) in ("running", "stalled")]
    if live:
        print("in progress (from the run records):")
        for rid, d in live:
            rec, st = RUN.record(d), RUN.state_of(d)
            what = "stalled" if st == "stalled" else ("running" if rec.get("status") == "running" else "waiting")
            print(f"{rid:12} {what:8} {(rec.get('started') or rec.get('created') or '')[5:16]:11} | heartbeat {RUN.heartbeat_age(d):.0f} s | {(rec.get('command') or '')[:70]}")


def missing(inside=lambda rid: True):
    """(id, run directory, state) of every run record that has ended (or whose launcher died: stalled) and has no ledger row, or only a planned / running /
    stalled one, or one of an EARLIER attempt (the record started later: a re-run whose own final ledger write failed)"""
    import run as RUN
    have = {r["id"]: r for r in load()}
    out = []
    for rid, d in run_dirs():
        st, old = RUN.state_of(d), have.get(rid, {})
        row = old.get("status")
        if inside(rid) and st in ("done", "ok", "failed", "timeout", "blocked", "gpu_busy", "stalled") \
                and (row in (None, "", "planned", "running") or (row == "stalled" and st != "stalled")  # a stalled row stays until the run really ends
                     or RUN.later_attempt(RUN.record(d), old)):
            out.append((rid, d, st))
    return out


def sync(dry=False):
    """add the ledger row of every ended run record that has none; the facts come from the record. A stalled record (the launcher died: WSL restart, kill)
    gets a `stalled` row, so the run is counted; the launcher's own final write replaces it if it ever finishes"""
    import run as RUN
    n = 0
    for rid, d, st in missing():
        rec = RUN.record(d)
        row = {c: rec.get(c, "") for c in COLS}
        row.update(id=rid, status="ok" if st == "done" else st, notes="row added by ledger.py sync from the run record"
                   + (f" (launcher gone, heartbeat {RUN.heartbeat_age(d) / 3600:.1f} h old)" if st == "stalled" else ""))
        if dry or RUN.ledger_write(P.path("ledger"), row, fill=True):  # the check is made again under the lock: a launcher that finished meanwhile wins
            print(f"{'would add' if dry else 'added'} {rid} ({row['status']})")
            n += 1
    print(f"{n} row(s) {'to add' if dry else 'added'}")
    if n and not dry:
        render()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "note", "add", "tail", "sync"])
    ap.add_argument("id", nargs="?")
    for c in ("hyp", "basis", "seed", "result", "decision", "notes", "status", "pred", "rule", "loop"):
        ap.add_argument("--" + c, default=None)
    ap.add_argument("-n", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cmd in ("note", "add") and not a.id:
        ap.error(f"{a.cmd} needs a run id (a row without an id would be written)")
    if a.cmd == "render":
        return render()
    if a.cmd == "tail":
        return tail(a.loop, a.n)
    if a.cmd == "sync":
        return sync(a.dry_run)
    with P.locked(P.path("ledger")):  # read, change and write as one step: two chats noting at once lose nothing
        rows = load()
        row = next((r for r in rows if r["id"] == a.id), None)
        if row is None:
            row = dict(id=a.id, status="planned")
            rows.append(row)
            if a.cmd == "note":  # a typo in an id would otherwise pass silently as a new planned row
                print(f"note: {a.id} had no ledger row; a new planned row was made (check the id)", file=sys.stderr)
        for k, col in (("hyp", "hypothesis"), ("basis", "basis"), ("seed", "seed"), ("result", "result"), ("decision", "decision"), ("notes", "notes"),
                       ("status", "status"), ("pred", "prediction"), ("rule", "rule")):
            if getattr(a, k) is not None:
                row[col] = getattr(a, k)
        save(rows)
    render()


if __name__ == "__main__":
    main()
