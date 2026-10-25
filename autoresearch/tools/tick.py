"""The tick of a loop: reconcile the run records, start the ready schedule items that fit the machine, report what runs. Standard library only.
Run it in the WSL, the same process namespace as the launchers (wsl python3 ../autoresearch/tools/tick.py ...): the GPU-slot owner check is by pid.

    python autoresearch/tools/tick.py [--dry-run] [--max-start N] [--handover]

One pass, idempotent and restartable (call it at every wake-up, from a timer, or by hand):
  1. reconcile: every schedule item with run ids takes its state from the run records (done / failed / stalled / running; schedule.py reconcile);
     a running item whose run has no record and no heartbeat (the launcher died before it wrote anything) becomes blocked
  2. start: ready items (state planned or ready, every dependency done) in priority order. Resource gpu needs a free GPU slot: project.json
     gpu_slots minus the busier of (slot directories held, schedule items running on the gpu whose record is not over); resource cpu runs at
     most cpu_slots (project.json, default 1) items at a time; resource agent (detail plan, code, summary) is never started by the tick - it
     needs a chat - and is listed as "needs the lead". An empty resource follows the kind (train -> gpu, evaluate -> cpu, else agent).
     GPU jobs started by hand without --gpu hold no slot: the tick cannot see them
  3. the command of an item: its `command` column, else the arm of the experiment file (run id E<id>.<arm> -> experiments/E<id>_*/experiment.json
     arms[<arm>].command). The card: the planned ledger row (ledger.py add) first, else the experiment file (title, prediction value vs arm,
     rule clauses); without a card run.py refuses and the item becomes blocked. TEST runs (experiment evaluate.split = test) are never started
     by the tick: they need the lead and a logged id.
     The item starts as `run.py --id <run id> [--gpu --wait-gpu 720] [--timeout <budget x timeout_factor>] -- <command>`, detached, its
     launcher output in <run_records>/<run id>/launcher.log. The tick waits (AR_ACCEPT_S, default 60 s) until the launcher has written its
     record (accepted) or exited (refused -> the row is blocked with the reason); the row is saved before and after every start
  4. status: running (heartbeat age), started now, waiting for a slot, needs the lead, blocked / failed, done - printed, and with --handover
     written between the markers <!-- tick --> and <!-- /tick --> of HANDOVER.md, each on its own line (added at the end when missing)
Budgets: "40m", "2h", "1.5 h", "90" (minutes); the timeout is budget x timeout_factor (project.json, default 2). Exit code 0; 1 when a start failed.
"""
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402
import run as RUN  # noqa: E402
import schedule as SC  # noqa: E402

POSIX = os.name == "posix"
ACCEPT_S = float(os.environ.get("AR_ACCEPT_S", "60"))
KIND_RESOURCE = {"train": "gpu", "evaluate": "cpu"}


def budget_minutes(text):
    """'40m' -> 40, '2h' -> 120, '1.5 h' -> 90, '90' -> 90, '' -> 0; None when the text cannot be read"""
    t = (text or "").strip().lower().replace(" ", "")
    if not t:
        return 0
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)(min|m|h|d)?", t)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2) or "m"
    return v * {"min": 1, "m": 1, "h": 60, "d": 1440}[u]


def resource_of(item):
    return item.get("resource") or KIND_RESOURCE.get(item.get("kind", ""), "agent")


def run_over(records, run_id):
    return RUN.state_of(os.path.join(records, run_id)) in ("done", "ok", "failed", "blocked", "gpu_busy", "timeout", "stalled")


def free_gpu_slots(records, rows, n):
    base = os.path.join(records, ".gpu_slots")
    probe = RUN.GpuSlot(records, n, "tick", "")
    held = sum(1 for i in range(n) if os.path.isdir(os.path.join(base, str(i))) and not probe.stale(os.path.join(base, str(i))))
    queued = sum(1 for r in rows if r["state"] == "running" and resource_of(r) == "gpu"
                 and not all(run_over(records, i) for i in (r.get("run_ids") or "").split(";") if i))
    return n - max(held, queued)


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
        arms = exp.get("arms")
        if isinstance(arms, dict) and arm in arms and isinstance(arms[arm], dict):
            return exp, arms[arm], f
    return None, None, None


def rule_text(exp):
    parts = []
    for c in exp.get("rule", []):
        if not isinstance(c, dict) or "metric" not in c:
            continue
        what = f">= +{c['min_effect']}" if "min_effect" in c else f">= -{c['max_loss']}" if "max_loss" in c else "?"
        subs = f" on {len(c['subsets'])} subsets (at least {c.get('at_least', len(c['subsets']))})" if isinstance(c.get("subsets"), list) else ""
        parts.append(f"{c['metric']} vs {c.get('vs', exp.get('control'))} {what}{subs}")
    return "; ".join(parts)


def launch_args(root, cfg, ledger, item, run_id):
    """the run.py arguments of a schedule item: (args before --, command list) or (None, reason)"""
    cmd = (item.get("command") or "").strip()
    args = ["--id", run_id]
    exp, arm, _ = experiment_arm(root, run_id)
    if exp is not None:
        if not cmd:
            cmd = (arm.get("command") or "").strip()
        if str(exp.get("evaluate", {}).get("split", "")).strip().lower() == str(cfg.get("holdout_split", "test")).lower():
            return None, f"holdout run ({cfg.get('holdout_split', 'test')}): needs the lead and a logged id (never started by the tick)"
        row = RUN.ledger_row(ledger, run_id)  # the planned row wins; the experiment file fills what it lacks
        pred = exp.get("prediction") if isinstance(exp.get("prediction"), dict) else {}
        pred_txt = f"{pred['value']} vs {pred['vs']} on {exp.get('metric', '')}" if "value" in pred and "vs" in pred else ""
        for flag, key, val in (("--hyp", "hypothesis", exp.get("title", "")), ("--pred", "prediction", pred_txt), ("--rule", "rule", rule_text(exp))):
            if not row.get(key) and val:
                args += [flag, val]
    if not cmd:
        return None, "no command (column `command` or arms[<arm>].command of the experiment file)"
    if resource_of(item) == "gpu":
        args += ["--gpu", "--wait-gpu", "720"]
    minutes = budget_minutes(item.get("budget"))
    if minutes is None:
        print(f"warning: budget '{item.get('budget')}' of {item['item']} not readable: no timeout")
    elif minutes:
        args += ["--timeout", str(minutes * float(cfg.get("timeout_factor", 2)))]
    return args, shlex.split(cmd, posix=POSIX)


def touch(path):
    try:
        os.utime(path, None)
    except OSError:
        pass


def start(root, records, run_id, args, cmd, lock):
    """start run.py detached and wait until it has accepted the run (record written) or refused; returns '' or the reason"""
    launcher = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), *args, "--", *cmd]
    d = os.path.join(records, run_id)
    os.makedirs(d, exist_ok=True)
    record = os.path.join(d, "run.json")
    before = os.path.getmtime(record) if os.path.exists(record) else None
    log = open(os.path.join(d, "launcher.log"), "a", encoding="utf-8")
    log.write(f"=== tick {RUN.now()}: {' '.join(launcher)}\n")
    log.flush()
    kw = dict(start_new_session=True) if POSIX else dict(creationflags=0x00000008 | 0x00000200)  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(launcher, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, AR_PROJECT=root), **kw)
    t0 = time.time()
    while True:
        if os.path.exists(record) and os.path.getmtime(record) != before:
            return ""  # accepted: run.py writes its record only after every refusal check
        if p.poll() is not None:
            if p.returncode == 0:
                return ""
            tail = open(os.path.join(d, "launcher.log"), encoding="utf-8", errors="replace").read().strip().splitlines()[-1:]
            return f"launcher exit {p.returncode}: {tail[0][:160] if tail else ''}"
        if time.time() - t0 > ACCEPT_S:
            return f"no run record after {ACCEPT_S:.0f} s (launcher pid {p.pid} still alive)"
        touch(lock)
        time.sleep(0.2)


def status_lines(rows, records, started, waiting, lead, slots_free):
    out = []
    for r in rows:
        if r["state"] != "running" or any(r is s for s, _ in started):
            continue
        ids = [x for x in (r.get("run_ids") or "").split(";") if x]
        ages = [(f"{i} heartbeat {RUN.heartbeat_age(os.path.join(records, i)):.0f} s" if RUN.heartbeat_age(os.path.join(records, i)) < 1e9 else f"{i} no heartbeat") for i in ids]
        out.append(f"running  {r['item']} ({resource_of(r)}, {r.get('budget', '')}): " + ", ".join(ages))
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


BLOCK = re.compile(r"^<!-- tick -->\n.*?^<!-- /tick -->$", re.M | re.S)


def write_handover(root, lines):
    p = os.path.join(root, "HANDOVER.md")
    with P.locked(p):
        text = open(p, encoding="utf-8").read() if os.path.exists(p) else "# HANDOVER\n"
        block = "<!-- tick -->\n" + f"tick {RUN.now()}:\n" + "\n".join("- " + ln for ln in lines) + "\n<!-- /tick -->"
        if BLOCK.search(text):
            text = BLOCK.sub(lambda m: block, text, count=1)
        else:
            text = text.rstrip("\n") + "\n\n## What runs (written by tick.py)\n" + block + "\n"
        P.atomic_write(p, text)


def tick(dry=False, max_start=10, handover=False):
    root, cfg = P.root(), P.cfg()
    records, ledger, sched = P.path("run_records"), P.path("ledger"), P.path("schedule")
    would = "would " if dry else ""
    with P.locked(sched):
        lock = sched + ".lock"
        rows = SC.load()
        changed = SC.reconcile(rows, records)
        for r in rows:  # a launcher that died before writing its record leaves nothing to reconcile from
            ids = [x for x in (r.get("run_ids") or "").split(";") if x]
            if r["state"] == "running" and ids and all(RUN.state_of(os.path.join(records, i)) == "" and RUN.heartbeat_age(os.path.join(records, i)) == float("inf") for i in ids):
                changed.append((r["item"], "running", "blocked"))
                r["state"], r["note"] = "blocked", "no run record and no heartbeat (the launcher left nothing)"
        started, waiting, lead, failed = [], [], [], 0
        gpu_free = free_gpu_slots(records, rows, int(cfg.get("gpu_slots", 2)))
        cpu_free = int(cfg.get("cpu_slots", 1)) - sum(1 for r in rows if r["state"] == "running" and resource_of(r) == "cpu")
        for r in SC.ready(rows):
            if len(started) >= max_start:
                waiting.append((r, "max-start reached"))
                continue
            res = resource_of(r)
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
            try:
                args, cmd = launch_args(root, cfg, ledger, r, run_id)
                if args is None:
                    if not dry:
                        r["state"], r["note"] = "blocked", cmd
                        SC.save(rows, lock=False)
                    print(f"{would}block {r['item']}: {cmd}")
                    failed += 1
                    continue
                if dry:
                    started.append((r, run_id))
                    continue
                r["state"], r["run_ids"] = "running", r.get("run_ids") or run_id  # saved before the start: a crash cannot start it twice
                SC.save(rows, lock=False)
                why = start(root, records, run_id, args, cmd, lock)
            except Exception as e:
                why = f"tick error: {type(e).__name__}: {str(e)[:120]}"
            if why and "still running" in why:
                why = ""  # a launcher of this id is alive: the row stays running
            if why:
                r["state"], r["note"] = "blocked", why
                SC.save(rows, lock=False)
                failed += 1
                continue
            SC.save(rows, lock=False)
            started.append((r, run_id))
            if res == "gpu":
                gpu_free -= 1
            else:
                cpu_free -= 1
            touch(lock)
        if not dry:
            SC.save(rows, lock=False)
    lines = status_lines(rows, records, started, waiting, lead, gpu_free)
    for item, old, new in changed:
        print(f"{would}reconcile {item}: {old} -> {new}")
    if dry and started:
        print("DRY RUN - would start: " + ", ".join(rid for _, rid in started))
    for ln in lines:
        print(("would " if dry and ln.startswith("started") else "") + ln)
    if handover and not dry:
        write_handover(root, lines)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="reconcile in memory and report; start nothing, write nothing")
    ap.add_argument("--max-start", type=int, default=10, help="start at most this many items per tick")
    ap.add_argument("--handover", action="store_true", help="write the status block into HANDOVER.md between <!-- tick --> markers")
    a = ap.parse_args()
    return tick(a.dry_run, a.max_start, a.handover)


if __name__ == "__main__":
    sys.exit(main())
