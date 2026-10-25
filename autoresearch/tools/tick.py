"""The tick of a loop: reconcile the run records, start the ready schedule items that fit the machine, report what runs. Standard library only.

    python autoresearch/tools/tick.py [--dry-run] [--max-start N] [--handover]

One pass, idempotent and restartable (call it at every wake-up, from a timer, or by hand):
  1. reconcile: every schedule item with run ids takes its state from the run records (done / failed / stalled / running; schedule.py reconcile)
  2. start: ready items (state planned or ready, every dependency done) in priority order. Resource gpu needs a free GPU slot (project.json
     gpu_slots, the slot directories of run.py); resource cpu runs at most cpu_slots (project.json, default 1) items at a time; resource agent
     (detail plan, code, summary) is never started by the tick - it needs a chat - and is listed as "needs the lead"
  3. the command of an item: its `command` column, else the arm of the experiment file (run id E<id>.<arm> -> experiments/E<id>_*/experiment.json
     arms[<arm>].command). The card (hypothesis, prediction, rule) comes from the experiment file, else run.py takes it from the planned ledger row.
     The item starts as `run.py --id <run id> [--gpu] [--timeout <budget>] -- <command>`, detached, its launcher output in
     <run_records>/<run id>/launcher.log; the schedule row becomes running. A launcher that refuses at once (no card, TEST, git identity)
     makes the row blocked with the reason in its note
  4. status: running (heartbeat age), started now, waiting for a slot, needs the lead, blocked / failed, done - printed, and with --handover
     written between the markers <!-- tick --> and <!-- /tick --> of HANDOVER.md (added at the end when missing): the "what runs" of the handover
Exit code 0; 1 when a start failed.
"""
import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402
import run as RUN  # noqa: E402
import schedule as SC  # noqa: E402

POSIX = os.name == "posix"


def budget_minutes(text):
    """'40m' -> 40, '2h' -> 120, '90' -> 90, '' -> 0"""
    t = (text or "").strip().lower()
    if not t:
        return 0
    try:
        if t.endswith("h"):
            return float(t[:-1]) * 60
        if t.endswith("m"):
            return float(t[:-1])
        return float(t)
    except ValueError:
        return 0


def free_gpu_slots(records, n):
    base = os.path.join(records, ".gpu_slots")
    probe = RUN.GpuSlot(records, n, "tick", "")
    busy = sum(1 for i in range(n) if os.path.isdir(os.path.join(base, str(i))) and not probe.stale(os.path.join(base, str(i))))
    return n - busy


def experiment_arm(root, run_id):
    """(experiment dict, arm dict, path) for a run id E<id>.<arm>, or (None, None, None)"""
    if "." not in run_id:
        return None, None, None
    eid, arm = run_id.split(".", 1)
    for f in sorted(glob.glob(os.path.join(root, "experiments", eid + "_*", "experiment.json"))):
        try:
            exp = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if arm in exp.get("arms", {}):
            return exp, exp["arms"][arm], f
    return None, None, None


def rule_text(exp):
    parts = []
    for c in exp.get("rule", []):
        if not isinstance(c, dict):
            continue
        what = f">= {c['min_effect']:+}" if "min_effect" in c else f">= -{c['max_loss']}" if "max_loss" in c else "?"
        parts.append(f"{c.get('metric')} vs {c.get('vs', exp.get('control'))} {what}" + (f" on {len(c['subsets'])} subsets (at least {c.get('at_least', len(c['subsets']))})" if c.get("subsets") else ""))
    return "; ".join(parts) or exp.get("kill", "")


def launch_args(root, item, run_id):
    """the run.py arguments of a schedule item: (args before --, command list) or (None, reason)"""
    cmd = (item.get("command") or "").strip()
    args = ["--id", run_id]
    exp, arm, _ = experiment_arm(root, run_id)
    if exp is not None:
        if not cmd:
            cmd = (arm.get("command") or "").strip()
        pred = exp.get("prediction", {})
        args += ["--hyp", exp.get("title", ""), "--pred", f"{pred.get('value', '')} vs {pred.get('vs', '')} on {exp.get('metric', '')}", "--rule", rule_text(exp)]
        if exp.get("evaluate", {}).get("split") == "test":
            args += ["--split", "test"]
    if not cmd:
        return None, "no command (column `command` or arms[<arm>].command of the experiment file)"
    if item.get("resource") == "gpu":
        args.append("--gpu")
    minutes = budget_minutes(item.get("budget"))
    if minutes:
        args += ["--timeout", str(minutes)]
    return args, shlex.split(cmd)


def start(root, records, item, run_id, args, cmd, dry):
    """start run.py detached; returns '' or the reason it did not start"""
    launcher = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), *args, "--", *cmd]
    if dry:
        return ""
    d = os.path.join(records, run_id)
    os.makedirs(d, exist_ok=True)
    log = open(os.path.join(d, "launcher.log"), "a", encoding="utf-8")
    log.write(f"=== tick {RUN.now()}: {' '.join(launcher)}\n")
    log.flush()
    kw = dict(start_new_session=True) if POSIX else dict(creationflags=0x00000008 | 0x00000200)  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(launcher, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, AR_PROJECT=root), **kw)
    time.sleep(2)
    if p.poll() is not None and p.returncode != 0:  # refused at once: no card, TEST not logged, git identity ...
        tail = open(os.path.join(d, "launcher.log"), encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
        return f"launcher exit {p.returncode}: {tail[0][:160] if tail else ''}"
    return ""


def status_lines(rows, records, started, waiting, lead, slots_free):
    out = []
    run = [r for r in rows if r["state"] == "running" and not any(r is s for s, _ in started)]  # started items are listed once, below
    for r in run:
        ids = [x for x in (r.get("run_ids") or "").split(";") if x]
        ages = [f"{i} heartbeat {RUN.heartbeat_age(os.path.join(records, i)):.0f} s" if RUN.heartbeat_age(os.path.join(records, i)) < 1e9 else f"{i} no heartbeat" for i in ids]
        out.append(f"running  {r['item']} ({r.get('resource', '')}, {r.get('budget', '')}): " + ", ".join(ages))
    for r, rid in started:
        out.append(f"started  {r['item']} -> {rid}")
    for r, why in waiting:
        out.append(f"waiting  {r['item']}: {why}")
    for r in lead:
        out.append(f"lead     {r['item']} ({r.get('kind', '')}): needs the lead chat")
    for r in rows:
        if r["state"] in ("blocked", "failed"):
            out.append(f"{r['state']:8} {r['item']}: {r.get('note', '')[:120]}")
    done = [r["item"] for r in rows if r["state"] == "done"]
    out.append(f"done     {len(done)}: " + ", ".join(done[-8:]) + ("" if len(done) <= 8 else " ..."))
    out.append(f"slots    gpu free {slots_free}")
    return out


def write_handover(root, lines):
    p = os.path.join(root, "HANDOVER.md")
    text = open(p, encoding="utf-8").read() if os.path.exists(p) else "# HANDOVER\n"
    block = "<!-- tick -->\n" + f"tick {RUN.now()}:\n" + "\n".join("- " + ln for ln in lines) + "\n<!-- /tick -->"
    if "<!-- tick -->" in text and "<!-- /tick -->" in text:
        a, b = text.index("<!-- tick -->"), text.index("<!-- /tick -->") + len("<!-- /tick -->")
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n## What runs (written by tick.py)\n" + block + "\n"
    P.atomic_write(p, text)


def tick(dry=False, max_start=10, handover=False):
    root, cfg = P.root(), P.cfg()
    records = P.path("run_records")
    with P.locked(P.path("schedule")):
        rows = SC.load()
        changed = SC.reconcile(rows, records)
        started, waiting, lead, failed = [], [], [], 0
        gpu_free = free_gpu_slots(records, int(cfg.get("gpu_slots", 2)))
        cpu_free = int(cfg.get("cpu_slots", 1)) - sum(1 for r in rows if r["state"] == "running" and r.get("resource") == "cpu")
        for r in SC.ready(rows):
            if len(started) >= max_start:
                waiting.append((r, "max-start reached"))
                continue
            res = r.get("resource") or "agent"
            if res == "agent":
                lead.append(r)
                continue
            if res == "gpu" and gpu_free <= 0:
                waiting.append((r, "no free GPU slot"))
                continue
            if res == "cpu" and cpu_free <= 0:
                waiting.append((r, "cpu slot busy"))
                continue
            run_id = (r.get("run_ids") or r["item"]).split(";")[0]
            args, cmd = launch_args(root, r, run_id)
            if args is None:
                r["state"], r["note"] = "blocked", cmd
                failed += 1
                continue
            why = start(root, records, r, run_id, args, cmd, dry)
            if why:
                r["state"], r["note"] = "blocked", why
                failed += 1
                continue
            if not dry:
                r["state"], r["run_ids"] = "running", r.get("run_ids") or run_id
            started.append((r, run_id))
            if res == "gpu":
                gpu_free -= 1
            else:
                cpu_free -= 1
        if not dry:
            SC.save(rows, lock=False)
    lines = status_lines(rows, records, started, waiting, lead, gpu_free)
    for item, old, new in changed:
        print(f"reconciled {item}: {old} -> {new}")
    print(("DRY RUN - would start: " + ", ".join(rid for _, rid in started)) if dry and started else "")
    for ln in lines:
        print(ln)
    if handover and not dry:
        write_handover(root, lines)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="reconcile and report, start nothing, write nothing")
    ap.add_argument("--max-start", type=int, default=10, help="start at most this many items per tick")
    ap.add_argument("--handover", action="store_true", help="write the status block into HANDOVER.md between <!-- tick --> markers")
    a = ap.parse_args()
    return tick(a.dry_run, a.max_start, a.handover)


if __name__ == "__main__":
    sys.exit(main())
