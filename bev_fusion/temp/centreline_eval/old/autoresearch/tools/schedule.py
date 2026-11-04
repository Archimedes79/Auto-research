"""The schedule of a loop: one row per work item. It says what happens next, so a disrupted loop resumes from it.

    python autoresearch/tools/schedule.py show                     # all items, grouped by state
    python autoresearch/tools/schedule.py next [--resource gpu]    # ready items: planned / ready, every dependency done
    python autoresearch/tools/schedule.py set E240.sum done        # set the stored state of an item that has no run (a review, a summary); `cancelled` also drops an item with runs
    python autoresearch/tools/schedule.py add E240.b --node R1 --kind train --depends E240.code --resource gpu --budget 40m --note "seed 1" [--command "bash tools/env/_py14.sh ..."]

Columns: item, node, kind, state, depends_on (';'-separated), resource (gpu / cpu / agent), budget, priority (1 = first), owner, run_ids (';'), note,
command (the launcher command of the item, for the lead to start through run.py). Items start through run.py (a chain
through --after); tick.py, which started ready items, was removed under D32 (it started 0 of 61 runs).

State is derived, never synced: an item with run ids takes its state from the run records (done flag, record status, heartbeat: derived() below), the
stored `state` column only counts for an item without a run record. Nothing writes a derived state back into the file, so the file cannot go stale.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

COLS = ["item", "node", "kind", "state", "depends_on", "resource", "budget", "priority", "owner", "run_ids", "note", "command"]
STATES = ("planned", "ready", "running", "done", "failed", "blocked", "cancelled")
OF_RUN = {"done": "done", "ok": "done", "running": "running", "failed": "failed", "gpu_busy": "failed", "timeout": "failed", "stalled": "failed", "blocked": "blocked", "stopped": "blocked", "held": "blocked"}   # stopped = ended on request with a checkpoint: not failed, but waits for a resume


def load(path=None):
    path = path or P.path("schedule")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows, path=None, lock=True):
    """whole-file write under the schedule lock with an atomic replace (lock=False when the caller holds P.locked already)"""
    import io
    path = path or P.path("schedule")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])
    if lock:
        with P.locked(path):
            P.atomic_write(path, buf.getvalue())
    else:
        P.atomic_write(path, buf.getvalue())


def derived(rows, records=None):
    """copies of the rows with the state taken from the run records wherever an item has run ids with a record; the input rows are not changed"""
    import run as RUN  # state_of: done flag, record status, heartbeat (stalled)
    records = records or P.path("run_records")
    out = []
    for r in rows:
        r = dict(r)
        ids = [x for x in (r.get("run_ids") or r["item"]).split(";") if x]  # the run id of an item without run_ids is its name
        st = [OF_RUN.get(RUN.state_of(os.path.join(records, i)), "") for i in ids]
        if any(st) and r["state"] != "cancelled":  # a cancelled item stays cancelled whatever its run records say
            r["state"] = "done" if all(s == "done" for s in st) else "failed" if "failed" in st else "blocked" if "blocked" in st else "running" if "running" in st else r["state"]
        out.append(r)
    return out


def _priority(r):
    """the priority of a row as a number (1 = first); an empty or hand-written value ('high') comes last instead of stopping `next` and the tick"""
    p = (r.get("priority") or "").strip()
    return int(p) if re.fullmatch(r"[+-]?[0-9]+", p) else 9


def ready(rows, resource=None):
    """ready items of derived rows, by priority"""
    done = {r["item"] for r in rows if r["state"] == "done"}
    out = [r for r in rows if r["state"] in ("planned", "ready") and all(d in done for d in (r.get("depends_on") or "").split(";") if d)
           and (resource is None or r.get("resource") == resource)]
    return sorted(out, key=lambda r: (_priority(r), rows.index(r)))


def change(a):
    """set / add: load, change and save as one step under the lock"""
    rows = load()
    if a.cmd == "set":
        if a.state not in STATES:
            sys.exit(f"state must be one of {STATES}")
        if not any(r["item"] == a.item for r in rows):
            sys.exit(f"item {a.item} is not in the schedule (`add` creates it)")
        for r in rows:
            if r["item"] == a.item:
                r["state"] = a.state
                shown = derived([r])[0]["state"]
                if shown != a.state:
                    print(f"note: {a.item} is shown as {shown}: its run record decides, the stored state counts only for an item without a record (cancel it with `set {a.item} cancelled`)")
    else:
        if any(r["item"] == a.item for r in rows):
            sys.exit(f"item {a.item} exists already in the schedule (change its state with `set`)")
        if a.priority and not re.fullmatch(r"[+-]?[0-9]+", a.priority):
            sys.exit(f"--priority must be a whole number (1 = first), got {a.priority!r}")
        res = a.resource or {"train": "gpu", "evaluate": "cpu"}.get(a.kind, "agent")  # an empty resource follows the kind
        rows.append(dict(item=a.item, node=a.node, kind=a.kind, state="planned", depends_on=a.depends, resource=res, budget=a.budget,
                         priority=a.priority or "5", owner=a.owner, run_ids=a.runs or (a.item if a.kind == "train" else ""), note=a.note, command=a.command))
    save(rows, lock=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "next", "set", "add"])
    ap.add_argument("item", nargs="?")
    ap.add_argument("state", nargs="?")
    ap.add_argument("--resource")
    for c in ("node", "kind", "depends", "budget", "priority", "owner", "runs", "note", "command"):
        ap.add_argument("--" + c, default="")
    a = ap.parse_args()
    if a.cmd in ("set", "add"):
        with P.locked(P.path("schedule")):
            return change(a)
    rows = derived(load())
    if a.cmd == "show":
        for s in STATES:
            sel = [r for r in rows if r["state"] == s]
            if sel:
                print(f"{s} ({len(sel)}): " + ", ".join(r["item"] for r in sel))
    else:
        for r in ready(rows, a.resource)[:10]:
            print(f"{r['item']:14} {r['kind']:12} {r['resource']:6} {r['budget']:6} {r['note']}")


if __name__ == "__main__":
    main()
