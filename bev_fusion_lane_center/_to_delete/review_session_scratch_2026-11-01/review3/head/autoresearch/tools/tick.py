"""The tick: read the schedule (item states derived from the run records), start the ready items that have a command and fit the machine, print what runs.
Standard library only. Run it in the WSL, the same process namespace as the launchers (wsl python3 ../autoresearch/tools/tick.py ...).

    python autoresearch/tools/tick.py [--dry-run] [--max-start N]

One pass, idempotent, safe to call at every wake-up. It writes nothing except a `blocked` state (with the reason) for an item whose launcher refused to start.
  1. derive: the state of an item with run ids comes from the run records (schedule.derived); nothing is synced or written back
  2. start: ready items (planned / ready, every dependency done) in priority order. A gpu item needs a free slot (project.json gpu_slots minus the slot
     directories that are held: the launchers' own semaphore, run.py GpuSlot); a launcher that still waits for its --after holds nothing. cpu items
     run at most cpu_slots (project.json, default 1) at a time. A real pass holds <run_records>/.tick, so a second tick exits at once and
     never starts an item twice; an item with several run ids starts the first one without a record. An agent item (detail plan, code, summary) is never started: it needs a chat. A TEST run (experiment
     evaluate.split = holdout split) is never started either. An item without a command is listed, not started.
  3. the command of an item: its `command` column, else the arm of the experiment file (run id E<id>.<arm> -> experiments/E<id>_*/experiment.json
     arms[<arm>].command). The card: the planned ledger row (ledger.py add) first, else the experiment file (title, prediction, rule clauses); without a
     card run.py refuses and the item becomes blocked. The item starts as `run.py --id <run id> [--gpu] [--timeout <budget x timeout_factor>] -- <command>`,
     detached, its launcher output in <run_records>/<run id>/launcher.log; the tick waits (AR_ACCEPT_S, default 60 s) until the launcher has written its
     record (accepted) or exited (refused -> blocked with the reason)
  4. status: running (heartbeat age), started now, waiting, needs the lead, needs a command, blocked / failed, done
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


def free_gpu_slots(records, n, base=None):
    """n minus the slot directories that are held by a live launcher (run.py GpuSlot is the semaphore; a stale directory does not count)"""
    base = base or os.path.join(records, ".gpu_slots")
    probe = RUN.GpuSlot(records, n, "tick", "", base=base)
    held = sum(1 for i in range(n) if os.path.isdir(os.path.join(base, str(i))) and not probe.stale(os.path.join(base, str(i))))
    return max(0, n - held)


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
        ev = exp.get("evaluate")
        if str((ev.get("split", "") if isinstance(ev, dict) else "")).strip().lower() == str(cfg.get("holdout_split", "test")).lower():
            return None, f"holdout run ({cfg.get('holdout_split', 'test')}): needs the lead and a logged id"
        row = RUN.ledger_row(ledger, run_id)  # the planned row wins; the experiment file fills what it lacks
        pred = exp.get("prediction") if isinstance(exp.get("prediction"), dict) else {}
        pred_txt = f"{pred['value']} vs {pred['vs']} on {exp.get('metric', '')}" if "value" in pred and "vs" in pred else ""
        for flag, key, val in (("--hyp", "hypothesis", exp.get("title", "")), ("--pred", "prediction", pred_txt), ("--rule", "rule", rule_text(exp))):
            if not row.get(key) and val:
                args += [flag, val]
    if not cmd:
        return None, "no command (column `command` or arms[<arm>].command of the experiment file)"
    if resource_of(item) == "gpu":
        args.append("--gpu")
    minutes = budget_minutes(item.get("budget"))
    if minutes is None:
        print(f"warning: budget '{item.get('budget')}' of {item['item']} not readable: no timeout")
    elif minutes:
        args += ["--timeout", str(minutes * float(cfg.get("timeout_factor", 2)))]
    return args, shlex.split(cmd, posix=POSIX)


def start(root, records, run_id, args, cmd):
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
        time.sleep(0.2)


def mark_blocked(item, why):
    """the one thing the tick writes: a blocked state with the reason, for an item whose launcher refused to start (it has no run record to derive from)"""
    with P.locked(P.path("schedule")):
        rows = SC.load()
        for r in rows:
            if r["item"] == item:
                r["state"], r["note"] = "blocked", why
        SC.save(rows, lock=False)


def status_lines(rows, records, started, waiting, lead, nocmd, blocked, free):
    out = []
    names = {r["item"] for r, _ in started}
    for r in rows:
        if r["state"] == "running" and r["item"] not in names:
            ids = [x for x in (r.get("run_ids") or r["item"]).split(";") if x]
            ages =[(f"{i} heartbeat {RUN.heartbeat_age(os.path.join(records, i)):.0f} s" if RUN.heartbeat_age(os.path.join(records, i)) < 1e9 else f"{i} no heartbeat") for i in ids]
            out.append(f"running  {r['item']} ({resource_of(r)}, {r.get('budget', '')}): " + ", ".join(ages))
    out += [f"started  {r['item']} -> {rid}" for r, rid in started]
    out += [f"waiting  {r['item']}: {why}" for r, why in waiting]
    out += [f"lead     {r['item']} ({r.get('kind', '')}): needs the lead chat" for r in lead]
    out += [f"command  {r['item']}: {why}" for r, why in nocmd]
    out += [f"blocked  {r['item']}: {why}" for r, why in blocked]
    out += [f"{r['state']:8} {r['item']}: {r.get('note', '')[:120]}" for r in rows if r["state"] in ("blocked", "failed") and r["item"] not in {b['item'] for b, _ in blocked}]
    done = [r["item"] for r in rows if r["state"] == "done"]
    out.append(f"done     {len(done)}: " + ", ".join(done[-8:]) + ("" if len(done) <= 8 else " ..."))
    out.append(f"slots    gpu free {free}")
    return out


def next_run_id(records, item):
    """the run id of an item that still has to start: the first of its run ids without a run record (an item `A;B` whose A is done starts B)"""
    ids = [x for x in (item.get("run_ids") or item["item"]).split(";") if x]
    return next((i for i in ids if not os.path.exists(os.path.join(records, i, "run.json"))), ids[0])


def tick(dry=False, max_start=10):
    """one pass; two ticks at once would start the same item twice, so a real pass holds <run_records>/.tick (a second tick exits at once)"""
    records = P.path("run_records")
    if dry:
        return tick_pass(True, max_start)
    os.makedirs(records, exist_ok=True)
    try:
        with P.locked(os.path.join(records, ".tick"), wait_s=0, stale_s=ACCEPT_S * max_start + 120):
            return tick_pass(False, max_start)
    except TimeoutError:
        print("another tick is running: nothing started")
        return 0


def tick_pass(dry, max_start):
    root, cfg = P.root(), P.cfg()
    records, ledger = P.path("run_records"), P.path("ledger")
    rows = SC.derived(SC.load())
    free = free_gpu_slots(records, int(cfg.get("gpu_slots", 2)), RUN.slot_dir(root, cfg, records))
    cpu_free = int(cfg.get("cpu_slots", 1)) - sum(1 for r in rows if r["state"] == "running" and resource_of(r) == "cpu")  # the box is shared: evaluations one at a time
    started, waiting, lead, nocmd, blocked = [], [], [], [], []
    for r in SC.ready(rows):
        res = resource_of(r)
        if res == "agent":
            lead.append(r)
            continue
        if len(started) >= max_start:
            waiting.append((r, "max-start reached"))
            continue
        if res == "gpu" and free <= 0:
            waiting.append((r, "no free GPU slot"))
            continue
        if res == "cpu" and cpu_free <= 0:
            waiting.append((r, "cpu slot busy"))
            continue
        run_id = next_run_id(records, r)
        try:
            args, cmd = launch_args(root, cfg, ledger, r, run_id)
            if args is None:
                (lead if cmd.startswith("holdout") else nocmd).append(r if cmd.startswith("holdout") else (r, cmd))
                continue
            why = "" if dry else start(root, records, run_id, args, cmd)
        except Exception as e:
            why = f"tick error: {type(e).__name__}: {str(e)[:120]}"
        if why:
            if not dry:  # a dry run writes nothing, also not on its error path
                try:
                    mark_blocked(r["item"], why)
                except Exception as e:  # a busy schedule must not end the pass after earlier items have started
                    why += f" (schedule not written: {type(e).__name__})"
            blocked.append((r, why))
            continue
        started.append((r, run_id))
        free -= res == "gpu"
        cpu_free -= res == "cpu"
    for line in status_lines(rows, records, started, waiting, lead, nocmd, blocked, max(0, free)):
        print("would start " + line[len("started "):] if dry and line.startswith("started ") else line)
    return 1 if blocked else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="derive and report; start nothing, write nothing")
    ap.add_argument("--max-start", type=int, default=10, help="start at most this many items per tick")
    a = ap.parse_args()
    return tick(a.dry_run, a.max_start)


if __name__ == "__main__":
    sys.exit(main())
