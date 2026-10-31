"""Experiment launcher: one command, one run record, one code tag.  Standard library only (works in any python of the WSL or Windows).

    python autoresearch/tools/run.py --id E240.a --hyp "..." --pred "Q >= 0.745" --rule "keep if > control + 0.010 on 2 seeds" --basis E219 --seed 0 \
           [--after E239.a,E239.c] [--gpu | --gpu-max-mb 9000] [--wait-gpu 720] [--timeout 180] [--retries 2] [--split test] [--result <metrics.json>] -- <command ...>

What it does (paths from the problem's project.json, see project.py):
  * run record  <run_records>/<id>/run.json  (command, start / end, host, driver, env, seed, pids, status, exit code, duration, commit, tag, last log lines)
  * full log    <run_records>/<id>/run.log   (stdout + stderr of the command); heartbeat file <run_records>/<id>/heartbeat touched every AR_HEARTBEAT_S
                (15) seconds while the launcher waits or the command runs
  * ledger row  <ledger>                     (one row per run id, written once when the run ends, under a lock with an atomic replace; render with ledger.py).
                A busy ledger is retried; if it stays busy the exit code stays the command's and `ledger.py sync` adds the row later from the run
                record. Runs in progress are not in the ledger: `ledger.py tail` and `schedule.py show` read them from the run records (state is
                derived from the records, never synced between files)
  * done flag   <run_records>/<id>/done      ONLY when the command exits with 0 (a crashed training never leaves a done flag)
  * card        a run needs its card: --hyp / --pred / --rule on the command line, else from the planned ledger row (ledger.py add); without
                one it is refused (--no-card for smoke tests and data preparation, the ledger row then says "no card")
  * code        snapshots the code folders (project.json code_dirs + fingerprint_extra + autoresearch/) through a TEMPORARY git index: a commit
                on top of HEAD that moves no branch and touches nobody's staging area, tagged <tag_prefix>/<id>. Refuses to start when git has no
                user identity (the snapshot would fail silently). --no-git skips this. The commit is written into the run record and the ledger.
                Not in the snapshot: data, checkpoints and the model registry unless listed in fingerprint_extra (the evaluated configuration is
                fixed by the metric card, which names the registry entry and the commit)
  * id reuse    an id that is still running (fresh heartbeat) or finished with exit 0 is refused (a new id, or --again for a deliberate repeat);
                a failed / blocked / stalled id is resumed: the previous log is kept as run.<n>.log, the first tag stays, the new attempt's commit
                and a note "code changed since attempt 1" (when the fingerprint differs) go into the record; result and decision of the earlier
                attempt are cleared in the ledger row when the new attempt starts; two launchers of one id starting at once: the second is refused
                (an atomic claim directory in the run directory)
  * after       --after <ids>: waits for the done flags of those runs (refuses when one of them failed or stalled); waiting is not counted as run time
  * holdout     --split <holdout> (project.json holdout_split, default test) is refused unless the run id appears in the problem's holdout log
                (project.json test_log)
  * chains      a launcher waits for its dependencies (--wait-after, default 720 min) and then for a GPU slot (--wait-gpu, default 720 min) before it
                gives up; give-up marks it blocked / gpu_busy and its own dependents are blocked in turn
  * GPU         --gpu: waits for a free GPU slot (project.json gpu_slots, default 2; slots are directories under project.json gpu_slots_dir
                (relative to the problem folder; problems that share one GPU name the same folder), default <run_records>/.gpu_slots, taken
                with an atomic mkdir; a slot whose owner is dead, whose run is over, or that has no owner file is reclaimed); --gpu-max-mb
                additionally waits until the GPU memory in use is below the limit; max --wait-gpu minutes (720), then status gpu_busy (exit 1). The
                owner check needs launcher and slot owner in the same process namespace (all WSL or all Windows)
  * timeout     --timeout <minutes>: the command and its process group are killed after the limit (status timeout, exit 124)
  * retry       a run whose LAST attempt printed a CUDA 'unknown error' is repeated (--retries) after 15 s; any other failure stops at once
  * stalled     a record in state running / waiting whose heartbeat is older than AR_STALE_S seconds (default 300) counts as stalled for --after,
                ledger.py tail and schedule.py show; records made before this version (no launcher_pid, no heartbeat file) count as running
Exit code = exit code of the command (1 if the GPU never became free, 2 if refused, 3 if a dependency failed, 124 on timeout).
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

COLS = ["id", "started", "ended", "status", "exit", "duration_s", "hypothesis", "basis", "seed", "command", "result", "decision", "notes", "prediction", "rule", "code_hash", "split", "commit"]
STALE_S = float(os.environ.get("AR_STALE_S", "300"))
HEARTBEAT_S = float(os.environ.get("AR_HEARTBEAT_S", "15"))
CUDA_RETRY = re.compile(r"CUDA (error: )?unknown error", re.I)
POSIX = os.name == "posix"


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def code_hash(root, cfg):
    """12-hex fingerprint of all .py / .sh of the code folders plus the extra files (works without git too)"""
    h = hashlib.sha1()
    for top in cfg["code_dirs"]:
        for dp, dn, fn in os.walk(os.path.join(root, top)):
            dn[:] = sorted(x for x in dn if x not in ("__pycache__", ".git"))
            for f in sorted(fn):
                if f.endswith((".py", ".sh")):
                    p = os.path.join(dp, f)
                    h.update(os.path.relpath(p, root).encode())
                    h.update(open(p, "rb").read())
    for extra in cfg["fingerprint_extra"]:
        p = os.path.join(root, extra)
        if os.path.isfile(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def git(root, *args, env=None):
    return subprocess.run(["git", "-c", "safe.directory=*", "-C", root, *args], capture_output=True, text=True, timeout=120, env=env)


def git_identity_error(root):
    """'' when git can commit here, else git's own message (no user.name / user.email, not a repository ...)"""
    if git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        return ""  # no repository: the snapshot is skipped, not an error
    r = git(root, "var", "GIT_COMMITTER_IDENT")
    return "" if r.returncode == 0 else (r.stderr.strip() or "git var GIT_COMMITTER_IDENT failed")[:160]


def git_snapshot(root, cfg, prefix, run_id, msg):
    """commit the code through a temporary index (shared index and branches untouched) and tag it prefix/run_id; returns (commit, tag, note)"""
    idx = None
    try:
        if git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            return "no-git", "", ""
        top = git(root, "rev-parse", "--show-toplevel").stdout.strip()
        tag = f"{prefix}/{run_id}"
        rel = os.path.relpath(root, top).replace("\\", "/")
        rel = "" if rel == "." else rel + "/"
        spec = [rel + d for d in cfg["code_dirs"]] + [rel + x for x in cfg["fingerprint_extra"]]
        ar = os.path.relpath(P.AR, top).replace("\\", "/")
        if not ar.startswith(".."):  # the research system itself, when it lives in the same repository
            spec.append(ar)
        head = git(top, "rev-parse", "-q", "--verify", "HEAD").stdout.strip()
        in_head = set(git(top, "ls-tree", "-r", "--name-only", "HEAD").stdout.split()) if head else set()
        spec = [s for s in spec if os.path.exists(os.path.join(top, s)) or any(h == s or h.startswith(s + "/") for h in in_head)]  # a deleted folder leaves the snapshot too
        fd, idx = tempfile.mkstemp(prefix="ar_index_")
        os.close(fd)
        os.remove(idx)
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        if head:
            r = git(top, "read-tree", "HEAD", env=env)
            if r.returncode != 0:
                return "git-error", "", "read-tree: " + r.stderr.strip()[:120]
        r = git(top, "add", "-A", "--", *spec, env=env)
        if r.returncode != 0:
            return "git-error", "", "add: " + r.stderr.strip()[:120]
        tree = git(top, "write-tree", env=env).stdout.strip()
        if head and tree == git(top, "rev-parse", "HEAD^{tree}").stdout.strip():
            commit = head
        else:
            args = ["commit-tree", tree, "-m", f"{run_id}: {msg}"[:200]] + (["-p", head] if head else [])
            r = git(top, *args, env=env)
            if r.returncode != 0:
                return "git-error", "", "commit-tree: " + r.stderr.strip()[:120]
            commit = r.stdout.strip()
        note = ""
        if git(top, "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0:
            first = git(top, "rev-list", "-n", "1", tag).stdout.strip()[:10]
            note = f"tag existed (resume): tag {tag} is attempt 1 ({first}), this attempt ran {commit[:10]}"
        else:
            for attempt in range(3):
                t = git(top, "tag", "-a", tag, "-m", msg[:200] or run_id, commit)
                if t.returncode == 0:
                    break
                time.sleep(2)
            else:
                return commit[:10], "", "tag failed: " + t.stderr.strip()[:120]
        return commit[:10], tag, note
    except Exception as e:
        return "git-error", "", repr(e)[:120]
    finally:
        if idx and os.path.exists(idx):
            os.remove(idx)


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split() if x.isdigit())
    except Exception:
        return None


def mem_ok(used_mb, max_mb):
    """the memory gate of --gpu-max-mb: no gate when max_mb is 0; a GPU that cannot be read (nvidia-smi failed: used_mb None) is NOT free"""
    return not max_mb or (used_mb is not None and used_mb < max_mb)


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return out.strip().splitlines()[0] if out.strip() else ""
    except Exception:
        return ""


def pid_alive(pid):
    """is a process with this pid alive? (on Windows os.kill(pid, 0) would TERMINATE it, so the process handle is queried instead)"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def heartbeat_age(d):
    try:
        return time.time() - os.path.getmtime(os.path.join(d, "heartbeat"))
    except OSError:
        return float("inf")


def record(d):
    """the run record of a directory, {} when there is none. A file that exists but does not parse, or a write in progress (a tmp file next to it),
    is retried a few times: on /mnt/c (9p) an atomic replace is not atomic for a concurrent reader (about 2 % torn / missing reads measured)"""
    path = os.path.join(d, "run.json")
    for i in range(4):
        try:
            return json.load(open(path))
        except Exception:
            try:
                writing = (os.path.exists(path) or any(n.startswith("run.json.tmp.") for n in os.listdir(d))
                           or time.time() - os.path.getmtime(d) < 1.0)         # the directory changed a moment ago: a replace may be in flight
            except OSError:
                writing = False
            if not writing or i == 3:
                return {}
            time.sleep(0.05)


def save_rec(path, rec, indent=1):
    """write a run record / slot owner file through a temporary file and an atomic replace: other tools poll these files (state_of, schedule,
    tick, GpuSlot.stale); a torn read gave record() == {}, and a live slot with an unreadable owner file looked stale and was taken twice"""
    text = json.dumps(rec, indent=indent)
    for i in range(3):
        try:
            return P.atomic_write(path, text)
        except PermissionError:      # native Windows: os.replace over a file that a reader has open
            time.sleep(0.1)
    P.atomic_write(path, text)


def state_of(d):
    """done | ok | failed | blocked | gpu_busy | timeout | running | stalled | '' from a run directory"""
    if os.path.exists(os.path.join(d, "done")):
        return "done"
    rec = record(d)
    st = rec.get("status", "")
    if st in ("running", "waiting"):
        if "launcher_pid" not in rec and not os.path.exists(os.path.join(d, "heartbeat")):
            return "running"  # a record of the launcher before heartbeats: nothing to judge it by
        return "stalled" if heartbeat_age(d) > STALE_S else "running"
    return st


def beat(hb, text=""):
    open(hb, "w").write(f"{now()} {text}".rstrip() + "\n")


def ledger_row(ledger, run_id):
    """the ledger row of an id (empty dict when none)"""
    if not os.path.exists(ledger):
        return {}
    with open(ledger, newline="", encoding="utf-8") as f:
        return next((r for r in csv.DictReader(f) if r["id"] == run_id), {})


def ledger_write(ledger, row, clear=(), fill=False):
    """merge one row into the ledger under the lock (a planned row keeps what was written before the run; `clear` names columns of the old row to drop).
    fill=True (ledger.py sync): only a missing row or a planned / running / stalled one is filled in, a finished row is never replaced; returns whether it wrote"""
    with P.locked(ledger):
        rows = []
        if os.path.exists(ledger):
            with open(ledger, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        old = next((r for r in rows if r["id"] == row["id"]), {})
        if fill and old.get("status") not in (None, "", "planned", "running", "stalled"):
            return False
        if fill and old.get("notes"):
            row = dict(row, notes=old["notes"] + "; " + row.get("notes", ""))
        old = {k: v for k, v in old.items() if k not in clear}
        row = {**old, **{k: v for k, v in row.items() if v not in ("", None)}}
        rows = [r for r in rows if r["id"] != row["id"]] + [row]
        os.makedirs(os.path.dirname(ledger), exist_ok=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")  # unknown keys never break the write (loop-18 lesson)
        w.writeheader()
        w.writerows([{c: r.get(c, "") for c in COLS} for r in rows])
        P.atomic_write(ledger, buf.getvalue())
        return True


def ledger_safe(ledger, row, clear=(), tries=3):
    """ledger_write that never kills the launcher: a busy ledger is retried; when it stays busy the run record keeps all facts and `ledger.py sync` adds the row later"""
    msg = ""
    for _ in range(tries):
        try:
            ledger_write(ledger, row, clear)
            return ""
        except Exception as e:
            msg = f"ledger not written ({type(e).__name__}: {str(e)[:120]}); the run record has the facts, `ledger.py sync` adds the row"
            time.sleep(2)
    print(f"[{row.get('id')}] {msg}")
    return msg


def wait_after(records, ids, minutes, hb=None):
    """wait for the done flags of ids; returns '' when all are done, else the reason; touches the heartbeat while waiting"""
    t0 = time.time()
    while True:
        if hb:
            beat(hb, "waiting for " + ",".join(ids))
        missing = []
        for i in ids:
            st = state_of(os.path.join(records, i))
            if st in ("done", "ok"):  # ok = exit 0 whose done flag could not be written
                continue
            if st in ("failed", "gpu_busy", "blocked", "timeout", "stalled"):
                return f"dependency {i} {st}"
            missing.append(i)
        if not missing:
            return ""
        if time.time() - t0 > minutes * 60:
            return f"dependencies not done after {minutes} min: {', '.join(missing)}"
        time.sleep(min(30, max(1, HEARTBEAT_S)))


def take_claim(d):
    """the right to start this run id: a directory made atomically in the run directory; a claim older than 60 s belongs to a launcher that died
    while starting and is taken over. None = another launcher holds it"""
    c = os.path.join(d, ".starting")
    try:
        os.mkdir(c)
    except FileExistsError:
        if time.time() - os.path.getmtime(c) < 60:
            return None
        os.utime(c)
    return c


class GpuSlot:
    """one of n GPU slots: a directory taken with an atomic mkdir; a slot whose owner is dead, whose run is over, or that has no owner file is reclaimed"""

    def __init__(self, records, n, run_id, run_dir, base=None):
        self.base, self.n, self.id, self.run_dir, self.held = base or os.path.join(records, ".gpu_slots"), n, run_id, run_dir, None

    def stale(self, s):
        if not os.path.exists(os.path.join(s, "run.json")):
            try:
                return time.time() - os.path.getmtime(s) > 10  # mkdir without the owner file: the taker died in between, or a leftover
            except OSError:
                return False
        o = record(s)
        if not o:                    # owner file there but unreadable right now: a taker that is writing it, not a dead one
            try:
                return time.time() - os.path.getmtime(s) > 10
            except OSError:
                return False
        owner = o.get("run_dir", "")
        return not pid_alive(o.get("pid")) or (owner and os.path.isdir(owner) and state_of(owner) != "running")

    def try_take(self):
        os.makedirs(self.base, exist_ok=True)
        for i in range(self.n):
            s = os.path.join(self.base, str(i))
            try:
                os.mkdir(s)
            except FileExistsError:
                if not self.stale(s):
                    continue
                shutil.rmtree(s, ignore_errors=True)
                try:
                    os.mkdir(s)
                except FileExistsError:
                    continue
            save_rec(os.path.join(s, "run.json"), dict(id=self.id, pid=os.getpid(), run_dir=self.run_dir, taken=now()), indent=None)
            self.held = s
            return True
        return False

    def release(self):
        if self.held:
            shutil.rmtree(self.held, ignore_errors=True)
            self.held = None


def slot_dir(root, cfg, records):
    """the folder of the GPU slots: project.json gpu_slots_dir (relative to the problem folder; shared by the problems of one machine), else per problem"""
    d = cfg.get("gpu_slots_dir")
    return os.path.normpath(os.path.join(root, d)) if d else os.path.join(records, ".gpu_slots")


def kill_tree(p):
    """kill the command and everything it started (its process group on POSIX, the process tree on Windows)"""
    try:
        if POSIX:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass
    p.wait()


def run_once(cmd, cwd, log_path, hb_path, timeout_s):
    """one attempt: returns (exit code, text the attempt wrote, pid, killed by the timeout); the heartbeat is touched while the command runs"""
    start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    with open(log_path, "a", encoding="utf-8", errors="replace") as lg:
        lg.write(f"=== attempt {now()}: {' '.join(cmd)}\n")
        lg.flush()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=lg, stderr=subprocess.STDOUT, start_new_session=POSIX)
        t0, last = time.time(), 0
        rc, timed_out = None, False
        while rc is None:
            if time.time() - last >= HEARTBEAT_S:
                beat(hb_path, f"pid {p.pid}")
                last = time.time()
            if timeout_s and time.time() - t0 > timeout_s:
                kill_tree(p)
                rc, timed_out = 124, True
                lg.write(f"=== killed after {timeout_s / 60:.0f} min (timeout)\n")
                break
            try:
                rc = p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                rc = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start)
        out = f.read()
    return rc, out, p.pid, timed_out


def main():
    root, cfg = P.root(), P.cfg()
    records, ledger, test_log = P.path("run_records"), P.path("ledger"), P.path("test_log")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="run id, e.g. E240.a (experiment id + letter for the arm / attempt)")
    ap.add_argument("--hyp", default="", help="one-sentence hypothesis")
    ap.add_argument("--pred", default="", help="prediction written BEFORE the run, a number (e.g. 'Q >= 0.745')")
    ap.add_argument("--rule", default="", help="decision rule written BEFORE the run")
    ap.add_argument("--split", default="val", help=f"the split or benchmark set of the run; the holdout split ('{cfg.get('holdout_split', 'test')}', project.json holdout_split) needs a row naming this id in the holdout log")
    card = ap.add_mutually_exclusive_group()
    card.add_argument("--require-card", action="store_true", help="(default) refuse to start without a card: --hyp, --pred and --rule, or a planned ledger row with them")
    card.add_argument("--no-card", action="store_true", help="start without a card (smoke tests, data preparation); the ledger row says 'no card'")
    ap.add_argument("--basis", default="")
    ap.add_argument("--seed", default="")
    ap.add_argument("--after", default="", help="comma list of run ids whose done flags this run waits for")
    ap.add_argument("--wait-after", type=int, default=720, help="minutes to wait for --after")
    ap.add_argument("--gpu", action="store_true", help="GPU job: start only when one of the GPU slots is free")
    ap.add_argument("--gpu-max-mb", type=int, default=0, help="GPU job: additionally wait until the GPU memory in use is below this (implies --gpu)")
    ap.add_argument("--gpu-slots", type=int, default=int(cfg.get("gpu_slots", 2)), help="number of GPU slots on this machine")
    ap.add_argument("--wait-gpu", type=int, default=720, help="minutes to wait for a GPU slot")
    ap.add_argument("--timeout", type=float, default=0, help="minutes after which the command is killed (0 = none)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--again", action="store_true", help="repeat an id that finished with exit 0 (otherwise refused)")
    ap.add_argument("--result", default="", help="json file the command writes its metrics to; embedded in the ledger when it exists")
    ap.add_argument("--no-git", action="store_true", help="do not snapshot and tag the code")
    ap.add_argument("--cwd", default=root, help="working directory of the command (default: problem folder)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command (put it after --)")
    planned = ledger_row(ledger, a.id)  # a row written before the run (ledger.py add) carries the card
    a.hyp, a.pred, a.rule = a.hyp or planned.get("hypothesis", ""), a.pred or planned.get("prediction", ""), a.rule or planned.get("rule", "")
    missing = [n for n, v in (("--hyp", a.hyp), ("--pred", a.pred), ("--rule", a.rule)) if not v]
    if missing:
        print(f"[{a.id}] card incomplete, missing {', '.join(missing)} (hypothesis, prediction and decision rule are written before the run: on the command line or in the planned ledger row)")
        if not a.no_card:
            print(f"[{a.id}] refused: a run needs its card (--no-card for a smoke test or data preparation)")
            return 2
    a.split = a.split.strip().lower()
    if cfg.get("splits") and a.split not in [s.lower() for s in cfg["splits"]]:
        print(f"[{a.id}] refused: split {a.split} is not one of the project's splits {cfg['splits']}")
        return 2
    if a.split == str(cfg.get("holdout_split", "test")).lower():
        log = open(test_log, encoding="utf-8").read() if os.path.exists(test_log) else ""
        if not re.search(r"(?<![\w.])" + re.escape(a.id) + r"(?![\w.])", log):  # the whole id, not a prefix of a longer id
            print(f"[{a.id}] refused: split={a.split} is the holdout split but {a.id} is not named in {os.path.relpath(test_log, root)} (log it first, with the person's OK)")
            return 2
    if not a.no_git:
        err = git_identity_error(root)
        if err:
            print(f"[{a.id}] refused: git cannot commit the snapshot here ({err}); set user.name / user.email in the repository, or --no-git")
            return 2
    d = os.path.join(records, a.id)
    os.makedirs(d, exist_ok=True)
    claim = take_claim(d)  # two launchers of one id (two ticks, two chats) must not both start: the first to make the directory goes on
    if claim is None:
        print(f"[{a.id}] refused: another launcher is starting this id right now")
        return 2
    notes = []
    try:
        prev = record(d)
        if prev:
            st = state_of(d)
            if st == "running":
                print(f"[{a.id}] refused: this id is still running (launcher pid {prev.get('launcher_pid')}, heartbeat {heartbeat_age(d):.0f} s old); use a new id")
                return 2
            if st in ("done", "ok") and not a.again:
                print(f"[{a.id}] refused: this id finished with exit 0 on {prev.get('ended')}; use a new id (the record would be overwritten) or --again")
                return 2
            notes.append(f"attempt after {st}")
        if os.path.exists(os.path.join(d, "done")):
            os.remove(os.path.join(d, "done"))
        n = 1
        while os.path.exists(os.path.join(d, f"run.{n}.log")):
            n += 1
        if os.path.exists(os.path.join(d, "run.log")):
            os.rename(os.path.join(d, "run.log"), os.path.join(d, f"run.{n}.log"))
        hb = os.path.join(d, "heartbeat")
        beat(hb, "created")
        again = bool(prev) or bool(ledger_row(ledger, a.id).get("started"))  # the id ran before (run record or ledger row)
        clear = ("result", "decision") if again else ()  # a new attempt starts without the earlier attempt's judgement
        rec = dict(id=a.id, hypothesis=a.hyp, prediction=a.pred, rule=a.rule, split=a.split, basis=a.basis, seed=a.seed, command=" ".join(cmd), cwd=a.cwd,
                   host=platform.node(), python=sys.version.split()[0], conda_env=os.environ.get("CONDA_DEFAULT_ENV", ""), gpu=gpu_info(), launcher_pid=os.getpid(),
                   created=now(), status="waiting", after=a.after, timeout_min=a.timeout)
        if prev.get("code_hash"):
            rec["attempts_before"] = prev.get("attempts", 0) + prev.get("attempts_before", 0)
        save_rec(os.path.join(d, "run.json"), rec)
    finally:
        try:
            os.rmdir(claim)
        except OSError:
            pass
    if again and ledger_row(ledger, a.id):
        ledger_safe(ledger, dict(id=a.id), clear)  # the old judgement goes now, not only at the end: a re-run that dies must not keep the old KEEP beside it
    if a.after:
        why = wait_after(records, [x for x in a.after.split(",") if x], a.wait_after, hb)
        if why:
            rec.update(status="blocked", ended=now(), notes=why)
            save_rec(os.path.join(d, "run.json"), rec)
            ledger_safe(ledger, dict(rec, exit=3, duration_s=0), clear)
            print(f"[{a.id}] blocked: {why}")
            return 3
    slot = GpuSlot(records, a.gpu_slots, a.id, d, base=slot_dir(root, cfg, records)) if (a.gpu or a.gpu_max_mb) else None
    try:
        if slot:
            t0 = time.time()
            while True:
                beat(hb, "waiting for the GPU")
                u = gpu_used_mb() if a.gpu_max_mb else None
                if mem_ok(u, a.gpu_max_mb) and slot.try_take():
                    break
                if time.time() - t0 > a.wait_gpu * 60:
                    rec.update(status="gpu_busy", ended=now(), notes=("nvidia-smi unreadable" if (a.gpu_max_mb and u is None) else f"GPU memory {u} MB in use") + f" or all {a.gpu_slots} slots taken for {a.wait_gpu} min")
                    save_rec(os.path.join(d, "run.json"), rec)
                    ledger_safe(ledger, dict(rec, exit=1, duration_s=0), clear)
                    print("GPU did not become free:", rec["notes"])
                    return 1
                time.sleep(min(30, max(1, HEARTBEAT_S)))
        ch = code_hash(root, cfg)
        if prev.get("code_hash") and prev["code_hash"] != ch:
            notes.append(f"code changed since attempt 1 ({prev['code_hash']} -> {ch})")
        commit, tag, gnote = ("skipped", "", "") if a.no_git else git_snapshot(root, cfg, cfg["tag_prefix"], a.id, a.hyp or " ".join(cmd))
        if gnote:
            notes.append(gnote)
        rec.update(status="running", started=now(), code_hash=ch, commit=commit, tag=tag or prev.get("tag", ""), git_note=gnote)
        save_rec(os.path.join(d, "run.json"), rec)
        t1 = time.time()
        rc, attempt, tail, timed_out = 1, 0, "", False
        while attempt <= a.retries:
            attempt += 1
            try:
                rc, out, pid, timed_out = run_once(cmd, a.cwd, os.path.join(d, "run.log"), hb, a.timeout * 60)
            except Exception as e:  # the command could not even be started (not found, bad cwd): a failed run, not a dead launcher
                rc, pid = 127, None
                out = f"launcher could not start the command: {type(e).__name__}: {e}"
                with open(os.path.join(d, "run.log"), "a", encoding="utf-8", errors="replace") as lg:
                    lg.write(out + "\n")
            rec["pid"] = pid
            tail = out[-3000:]
            if rc == 0 or timed_out or not CUDA_RETRY.search(out):
                break
            if attempt <= a.retries:
                time.sleep(15)
        dur = round(time.time() - t1)
    finally:
        if slot:
            slot.release()
    res = ""
    if a.result and os.path.exists(a.result):
        try:
            res = json.dumps(json.load(open(a.result)))[:400]
        except Exception:
            res = "(result file unreadable)"
    status = "ok" if rc == 0 else "timeout" if timed_out else "failed"  # a command's own exit 124 is a failure, not the launcher's timeout
    rec.update(status=status, exit=rc, duration_s=dur, attempts=attempt, ended=now(), last_lines=tail.strip().splitlines()[-12:])
    try:
        save_rec(os.path.join(d, "run.json"), rec)
    except OSError as e:             # the command has finished: the done flag and the ledger row matter more than this file
        print(f"[{a.id}] warning: run record not written ({e!r})", flush=True)
    if rc == 0:
        open(os.path.join(d, "done"), "w").write(rec["ended"] + "\n")
    if attempt > 1:
        notes.insert(0, f"{attempt} attempt(s)")
    if a.no_card and missing:
        notes.append("no card")
    ledger_safe(ledger, dict(rec, result=res, notes="; ".join(notes)), clear)
    print(f"[{a.id}] {status} (exit {rc}, {dur} s, {attempt} attempt(s)), code {commit} {rec['tag']}; log {os.path.join(d, 'run.log')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
