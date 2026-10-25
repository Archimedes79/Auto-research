"""The schedule of a loop: one row per work item (columns below). It says what happens next, so a disrupted loop resumes from it.

    python autoresearch/tools/schedule.py show                     # all items, grouped by state
    python autoresearch/tools/schedule.py next [--resource gpu]    # ready items: state planned/ready and every dependency done
    python autoresearch/tools/schedule.py set E240.a running       # change a state (planned, ready, running, done, failed, blocked, cancelled)
    python autoresearch/tools/schedule.py reconcile                # items whose run ids have run records: running / done / failed from the records
    python autoresearch/tools/schedule.py add E240.b --node R1 --kind train --depends E240.code --resource gpu --budget 40m --note "seed 1"

Columns: item, node, kind, state, depends_on (';'-separated), resource (gpu / cpu / agent), budget, priority (1 = first), owner, run_ids (';'), note.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

COLS = ["item", "node", "kind", "state", "depends_on", "resource", "budget", "priority", "owner", "run_ids", "note"]
STATES = ("planned", "ready", "running", "done", "failed", "blocked", "cancelled")


def load(path=None):
    path = path or P.path("schedule")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows, path=None):
    """whole-file write under the schedule lock with an atomic replace"""
    import io
    path = path or P.path("schedule")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])
    with P.locked(path):
        P.atomic_write(path, buf.getvalue())


def ready(rows, resource=None):
    done = {r["item"] for r in rows if r["state"] == "done"}
    out = [r for r in rows if r["state"] in ("planned", "ready") and all(d in done for d in (r.get("depends_on") or "").split(";") if d)
           and (resource is None or r.get("resource") == resource)]
    return sorted(out, key=lambda r: (int(r.get("priority") or 9), rows.index(r)))


def reconcile(rows, records=None):
    records = records or P.path("run_records")
    changed = []
    for r in rows:
        ids = [x for x in (r.get("run_ids") or "").split(";") if x]
        if not ids or r["state"] in ("done", "cancelled"):
            continue
        import run as RUN  # state_of: done flag, status, heartbeat (stalled)
        st = [{"done": "done", "ok": "done", "running": "running", "failed": "failed", "gpu_busy": "failed", "timeout": "failed", "stalled": "failed", "blocked": "blocked"}.get(RUN.state_of(os.path.join(records, i)), "")
              for i in ids]
        new = "done" if st and all(s == "done" for s in st) else "failed" if "failed" in st else "blocked" if "blocked" in st else "running" if "running" in st else r["state"]
        if new != r["state"]:
            changed.append((r["item"], r["state"], new))
            r["state"] = new
    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "next", "set", "reconcile", "add"])
    ap.add_argument("item", nargs="?")
    ap.add_argument("state", nargs="?")
    ap.add_argument("--resource")
    for c in ("node", "kind", "depends", "budget", "priority", "owner", "runs", "note"):
        ap.add_argument("--" + c, default="")
    a = ap.parse_args()
    rows = load()
    if a.cmd == "show":
        for s in STATES:
            sel = [r for r in rows if r["state"] == s]
            if sel:
                print(f"{s} ({len(sel)}): " + ", ".join(r["item"] for r in sel))
    elif a.cmd == "next":
        for r in ready(rows, a.resource)[:10]:
            print(f"{r['item']:14} {r['kind']:12} {r['resource']:6} {r['budget']:6} {r['note']}")
    elif a.cmd == "set":
        if a.state not in STATES:
            sys.exit(f"state must be one of {STATES}")
        for r in rows:
            if r["item"] == a.item:
                r["state"] = a.state
        save(rows)
    elif a.cmd == "reconcile":
        for item, old, new in reconcile(rows):
            print(f"{item}: {old} -> {new}")
        save(rows)
    elif a.cmd == "add":
        rows.append(dict(item=a.item, node=a.node, kind=a.kind, state="planned", depends_on=a.depends, resource=a.resource or "", budget=a.budget,
                         priority=a.priority or "5", owner=a.owner, run_ids=a.runs or (a.item if a.kind == "train" else ""), note=a.note))
        save(rows)


if __name__ == "__main__":
    main()
