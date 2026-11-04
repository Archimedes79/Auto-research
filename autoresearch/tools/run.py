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
  * stop        exit code 75 (trainsuite: the trainer was asked to stop and saved a checkpoint) is status "stopped": no done flag, no retry (not even when the log tail
                holds a CUDA error of an earlier attempt), not counted as a failure in metrics.py; the id is put ON HOLD (a HOLD file in its run record, hold.py): it is
                refused ("held") and starts again only after the person's words (hold.py clear ID --words ...); --after dependents are blocked. The hold is read
                again every round while the launcher waits (--after, machine lock, GPU queue): a hold set meanwhile ends the wait (status held, exit 2); an id whose run
                folder is the same after . -> _ (`t1.a`, `t1_a`) is refused too
  * card        a run needs its card: --hyp / --pred / --rule on the command line, else from the planned ledger row (ledger.py add); without
                one it is refused (--no-card for smoke tests and data preparation, the ledger row then says "no card")
  * code        snapshots the code folders (project.json code_dirs + fingerprint_extra + autoresearch/) through a TEMPORARY git index: a commit
                on top of HEAD that moves no branch and touches nobody's staging area, tagged <tag_prefix>/<id>. Refuses to start when git has no
                user identity (the snapshot would fail silently). --no-git skips this. The commit is written into the run record and the ledger.
                --pin REF runs on the code of that commit or tag instead (a resume that continues on the code of attempt 1: `trainsuite queue resume`): no new
                commit or tag, no drift check, a note in the record. Not in the snapshot: data, checkpoints and the model registry unless listed in fingerprint_extra (the evaluated configuration is
                fixed by the metric card, which names the registry entry and the commit). Guaranteed: the commit names the code at LAUNCH. That the
                command ran it is the command's job (it runs `git archive $AR_COMMIT`); every other command runs from the working tree. So when the
                run ends the code files on disk (.py / .sh of the code folders and of autoresearch/, fingerprint_extra) are looked at again: a
                hash like the launch one (about a second) says whether anything changed; only then are they compared with the snapshot (the paths
                of the launch, a temporary index): a difference is written into the run record (code_drift: n, paths = the first 10) and as a note
                into the ledger row; no difference leaves no key, a check that itself fails leaves code_drift_check = "failed: <reason>" and never
                fails the run; --no-drift-check skips it. Not seen: an edit made between the first hash and the snapshot and undone before the end
  * id reuse    an id that is still running (fresh heartbeat) or finished with exit 0 is refused (a new id, or --again for a deliberate repeat);
                a failed / blocked / stalled / stopped id is resumed: the previous log is kept as run.<n>.log, the first tag stays, the new attempt's commit
                and a note "code changed since attempt 1" (when the fingerprint differs) go into the record; result and decision of the earlier
                attempt are cleared in the ledger row when the new attempt starts; two launchers of one id starting at once: the second is refused
                (an atomic claim directory in the run directory)
  * after       --after <ids>: waits for the done flags of those runs (refuses when one of them failed, stalled or was stopped); waiting is not counted as run time
  * holdout     --split <holdout> (project.json holdout_split, default test) is refused unless the run id appears in the problem's holdout log
                (project.json test_log)
  * chains      a launcher waits for its dependencies (--wait-after, default 720 min) and then for a GPU slot (--wait-gpu, default 720 min) before it
                gives up; give-up marks it blocked / gpu_busy and its own dependents are blocked in turn
  * GPU         --gpu: waits for a free GPU slot (project.json gpu_slots, default 2; slots are directories under project.json gpu_slots_dir
                (relative to the problem folder; problems that share one GPU name the same folder AND the same gpu_slots count), default
                <run_records>/.gpu_slots, taken
                with an atomic mkdir; a slot whose owner is dead, whose run is over, or that has no owner file is reclaimed, one launcher at a time
                (lock <slots dir>/.reclaim.lock); the owner file holds a token and a launcher releases only a slot that still carries its own); --gpu-max-mb
                additionally waits until the GPU memory in use is below the limit; max --wait-gpu minutes (720), then status gpu_busy (exit 1). The
                owner check needs launcher and slot owner in the same process namespace (all WSL or all Windows)
  * queue / RAM the GPU jobs that wait stand in ONE line (tickets under <slots dir>/.queue, shared by the problems that share the GPU): first come, first
                served, a ticket of a dead launcher does not count, the place is given up when the run starts. --ram-mb N (project.json ram_mb, 0 = off): the run
                starts only when /proc/meminfo MemAvailable is at least N on AR_RAM_CHECKS (3) looks in a row, so two jobs run in parallel exactly when the RAM is
                there for both and the box does not swap (a training with 6 loader workers wants about 10 GB of the 24); the heartbeat says what it waits for.
                Settle time (--settle-s, project.json gpu_settle_s, 0 = off; bev_fusion 180; AR_SETTLE_S overrides): a launcher waits while another one took its slot less
                than that many seconds ago, because that run has not loaded its data or allocated its GPU memory yet and the RAM / memory gates would read the box too early
                (two trainings that started in the same second slowed each other 13 times); it delays only the launchers of a project that sets it. A ticket whose
                launcher the machine lock holds back does not hold up the line
  * machine lock (D19) every run, GPU or not, waits (max --wait-gpu minutes, then gpu_busy) while another owner holds a live exclusive lock of
                machine_lock.py; the launcher's own owner name is AR_LOCK_OWNER from its environment (the lock owner's own runs pass, also when a launcher
                that the lock holds back stands earlier in the line)
  * environment the command gets AR_RUN_ID and, with a snapshot, AR_COMMIT / AR_TAG (the commit this attempt was launched from; the tag names attempt 1)
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
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hold as HOLD  # noqa: E402
import machine_lock  # noqa: E402
import project as P  # noqa: E402

COLS = ["id", "started", "ended", "status", "exit", "duration_s", "hypothesis", "basis", "seed", "command", "result", "decision", "notes", "prediction", "rule", "code_hash", "split", "commit"]
STALE_S = float(os.environ.get("AR_STALE_S", "300"))
HEARTBEAT_S = float(os.environ.get("AR_HEARTBEAT_S", "15"))
RAM_CHECKS = int(os.environ.get("AR_RAM_CHECKS", "3"))     # looks in a row (one per wait round) with enough free RAM before a run starts
CUDA_RETRY = re.compile(r"CUDA (error: )?unknown error", re.I)
SCRATCH_MAX_BYTES = 1_000_000   # a scratch script over this size is not put into the run tag (it is data, not a script)
SCRATCH_MAX_FILES = 2000        # at most this many scratch scripts per tag (about 10 ms each at the snapshot, and again at the drift check)
SCRATCH_SKIPPED = [0]           # how many scripts the last scratch_scripts() call left out
EXIT_STOPPED = 75   # trainsuite/control.py EXIT_STOPPED: a trainer that was asked to stop (CONTROL file) saved a checkpoint and ended on purpose: status "stopped", never retried
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


def snapshot_spec(root, cfg):
    """(repository top, the paths a snapshot covers relative to it, HEAD or ''): code_dirs, fingerprint_extra and the research system itself"""
    top = git(root, "rev-parse", "--show-toplevel").stdout.strip()
    rel = os.path.relpath(root, top).replace("\\", "/")
    rel = "" if rel == "." else rel + "/"
    # normalised: a code folder outside the problem folder (code_dirs "../autogt") becomes the repository path "autogt"
    spec = [posixpath.normpath(rel + d) for d in cfg["code_dirs"]] + [posixpath.normpath(rel + x) for x in cfg["fingerprint_extra"]]
    ar = os.path.relpath(P.AR, top).replace("\\", "/")
    if not ar.startswith(".."):  # the research system itself, when it lives in the same repository
        spec.append(ar)
    head = git(top, "rev-parse", "-q", "--verify", "HEAD").stdout.strip()
    in_head = set(git(top, "ls-tree", "-r", "--name-only", "HEAD").stdout.split()) if head else set()
    spec = [s for s in spec if os.path.exists(os.path.join(top, s)) or any(h == s or h.startswith(s + "/") for h in in_head)]  # a deleted folder leaves the snapshot too
    return top, spec, head


def scratch_scripts(top, root, cfg):
    """the .py / .sh files below the scratch folder (project.json scratch, default "temp"), relative to the repository top; [] when the project sets it empty. The
    .gitignore keeps scratch out of ordinary commits (check_git.py), so the snapshot adds these by force: a run's tag (and `clean_temp.py --move`, which tags
    the same way) still holds the scripts of a loop. This holds also when the scratch folder lies OUTSIDE the code folders (bev_fusion: code_dirs training /
    evaluation / autogt / tests, scratch temp): its scripts are in the tag but not in the drift check, which watches the code folders only, so another chat's new
    scratch script is no 'code drift' of a run. A script over SCRATCH_MAX_BYTES and everything beyond the first SCRATCH_MAX_FILES (sorted by path) is left out;
    the number is in SCRATCH_SKIPPED[0] and git_snapshot names it in the note of the run"""
    norm = lambda x: posixpath.normpath(x.replace("\\", "/")).strip("/")
    s = norm(cfg.get("scratch") or "")
    if not s or s == ".":
        return []
    out, skipped = [], 0
    for dp, dn, fns in os.walk(os.path.join(root, s)):
        dn[:] = [d for d in dn if d != "__pycache__" and not d.startswith("literature_backup_")]  # never in git (the old .gitignore rule)
        for fn in fns:
            if fn.endswith((".py", ".sh")):
                p = os.path.join(dp, fn)
                try:
                    big = os.path.getsize(p) > SCRATCH_MAX_BYTES
                except OSError:                       # gone while walking
                    continue
                if big:
                    skipped += 1
                else:
                    out.append(os.path.relpath(p, top).replace("\\", "/"))
    out.sort()
    if len(out) > SCRATCH_MAX_FILES:                  # a cloned repository or a site-packages copy in the scratch folder must not slow every launch and fill every tag
        skipped += len(out) - SCRATCH_MAX_FILES
        out = out[:SCRATCH_MAX_FILES]
    SCRATCH_SKIPPED[0] = skipped
    return out


def write_tree(top, spec, head, force=None):
    """tree id of HEAD with the spec paths replaced by the working tree, made in a TEMPORARY index (the shared index is untouched); returns (tree, error text).
    A spec path that is a file (fingerprint_extra) is added even when a .gitignore rule matches it (named on purpose). `force` = a function that returns more
    files to add by force (the scratch scripts); it is called AFTER the slow `add -A`, so a script that is moved while the index is built is not named, and
    a file that is gone at the add is skipped"""
    fd, idx = tempfile.mkstemp(prefix="ar_index_")
    os.close(fd)
    os.remove(idx)
    try:
        env = dict(os.environ, GIT_INDEX_FILE=idx)
        if head:
            r = git(top, "read-tree", "HEAD", env=env)
            if r.returncode != 0:
                return "", "read-tree: " + r.stderr.strip()[:120]
        files = [s for s in spec if os.path.isfile(os.path.join(top, s))]
        dirs = [s for s in spec if s not in files]  # a folder, or a file that is gone from disk but still in HEAD (its deletion is recorded)
        for args in ((("-A",), dirs), (("-f", "-A"), files)):
            if args[1]:
                r = git(top, "add", *args[0], "--", *args[1], env=env)
                if r.returncode != 0:
                    return "", "add: " + r.stderr.strip()[:120]
        lit = dict(env, GIT_LITERAL_PATHSPECS="1")  # a name with [ or * is no pattern
        more = [f for f in (force() if force else []) if os.path.lexists(os.path.join(top, f))]
        for i in range(0, len(more), 100):  # chunks keep the command line short
            chunk = more[i:i + 100]
            if git(top, "add", "-f", "--", *chunk, env=lit).returncode != 0:  # one file went between the check and the add: one by one, the gone ones are skipped
                for f in chunk:
                    r = git(top, "add", "-f", "--", f, env=lit)
                    if r.returncode != 0 and os.path.lexists(os.path.join(top, f)):
                        return "", "add -f: " + r.stderr.strip()[:120]
        r = git(top, "write-tree", env=env)
        if r.returncode != 0:
            return "", "write-tree: " + r.stderr.strip()[:120]
        return r.stdout.strip(), ""
    finally:
        if os.path.exists(idx):
            os.remove(idx)


def git_snapshot(root, cfg, prefix, run_id, msg):
    """commit the code through a temporary index (shared index and branches untouched) and tag it prefix/run_id; returns (commit, tag, note)"""
    try:
        if git(root, "rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            return "no-git", "", ""
        tag = f"{prefix}/{run_id}"
        top, spec, head = snapshot_spec(root, cfg)
        SCRATCH_SKIPPED[0] = 0
        tree, err = write_tree(top, spec, head, lambda: scratch_scripts(top, root, cfg))
        if err:
            return "git-error", "", err
        if head and tree == git(top, "rev-parse", "HEAD^{tree}").stdout.strip():
            commit = head
        else:
            args = ["commit-tree", tree, "-m", f"{run_id}: {msg}"[:200]] + (["-p", head] if head else [])
            r = git(top, *args)
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
        if SCRATCH_SKIPPED[0]:
            note = "; ".join(x for x in (note, f"{SCRATCH_SKIPPED[0]} scratch script(s) not in the tag (over {SCRATCH_MAX_BYTES // 1000} kB, or beyond the first {SCRATCH_MAX_FILES})") if x)
        return commit[:10], tag, note
    except Exception as e:
        return "git-error", "", repr(e)[:120]


def pin_snapshot(root, ref, prev):
    """(commit, tag, note) of a --pin run: the commit that `ref` names (the tag of attempt 1, or any commit), nothing new is committed or tagged; the tag stays attempt 1's"""
    try:
        r = git(root, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}")
    except Exception as e:
        return "git-error", "", repr(e)[:120]
    commit = r.stdout.strip()
    if r.returncode != 0 or not commit:
        return "git-error", "", f"--pin {ref}: not a commit of this repository"
    return commit[:10], prev.get("tag", ""), f"pinned: this attempt ran the code of {ref} ({commit[:10]}), not a snapshot of the working tree"


def drift_hash(root, cfg):
    """code_hash over the code folders AND the research system's own .py / .sh: the cheap first look of the drift check"""
    return code_hash(root, dict(cfg, code_dirs=[*cfg["code_dirs"], os.path.relpath(P.AR, root)]))


def code_drift(root, cfg, commit, before, spec):
    """did the code files on disk change while the run went? before = drift_hash and spec = the snapshot's paths, both taken at launch; commit = the
    snapshot. The same hash now -> ({}, '') at once. Else the working tree (temporary index) is compared with the snapshot over the LAUNCH paths (HEAD
    may have moved on, a folder may be gone), keeping the code files (.py / .sh, fingerprint_extra): ({'n': differing files, 'paths': the first 10}, '')
    or ({}, '') without one; a check that fails gives ({}, 'failed: <reason>'), it never raises"""
    try:
        if spec is None or before is None:
            return {}, "failed: no launch state (snapshot paths or hash)"
        if drift_hash(root, cfg) == before:
            return {}, ""
        top, spec_now, head = snapshot_spec(root, cfg)
        live, err = write_tree(top, spec_now, head, lambda: scratch_scripts(top, root, cfg))
        if err:
            return {}, "failed: " + err
        r = git(top, "diff-tree", "-r", "--name-only", commit, live, "--", *spec)
        if r.returncode != 0:
            return {}, "failed: diff-tree: " + r.stderr.strip()[:100]
        paths = [p for p in r.stdout.splitlines() if p.endswith((".py", ".sh")) or any(p.endswith(x) for x in cfg["fingerprint_extra"])]
        return ({"n": len(paths), "paths": paths[:10]} if paths else {}), ""
    except Exception as e:
        return {}, "failed: " + repr(e)[:100]


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return max(int(x) for x in out.split() if x.isdigit())
    except Exception:
        return None


def mem_ok(used_mb, max_mb):
    """the memory gate of --gpu-max-mb: no gate when max_mb is 0; a GPU that cannot be read (nvidia-smi failed: used_mb None) is NOT free"""
    return not max_mb or (used_mb is not None and used_mb < max_mb)


def mem_available_mb():
    """MemAvailable of /proc/meminfo (or of the file named by AR_MEMINFO) in MB; None when it cannot be read (Windows): no RAM gate then"""
    try:
        with open(os.environ.get("AR_MEMINFO", "/proc/meminfo"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class RamGate:
    """--ram-mb: a run starts only when at least `need_mb` of RAM are available on RAM_CHECKS looks in a row. The box swaps when two trainings and a selftest share
    24 GB (a training with 6 loader workers wants about 10 GB): a second job then makes both slower than one after the other. Two jobs run in parallel exactly when
    the RAM is there for both"""

    def __init__(self, need_mb):
        self.need, self.n, self.last = need_mb, 0, None

    def passed(self):
        self.last = mem_available_mb()
        if self.last is None:
            return True
        self.n = self.n + 1 if self.last >= self.need else 0
        return self.n >= RAM_CHECKS

    def reset(self):
        self.n = 0


def settle_s(cfg):
    """seconds a slot counts as `loading` after it was taken: project.json gpu_settle_s (0 = off); AR_SETTLE_S overrides it (the selftests)"""
    try:
        return float(os.environ.get("AR_SETTLE_S", cfg.get("gpu_settle_s", 0)))
    except (TypeError, ValueError):
        return 0.0


def blocked_by_lock(owner):
    """does the live exclusive machine lock of another owner hold back a launcher of this lock owner? (the lock must never be what breaks a launch: a failure says no)"""
    try:
        return machine_lock.blocks(owner) is not None
    except Exception:  # noqa: BLE001
        return False


class Ticket:
    """a place in the line of the launchers that wait for a GPU slot (and RAM): first come, first served, so the run that has waited longest starts first. A ticket
    whose launcher is dead does not count and is removed; a ticket whose launcher the exclusive machine lock of somebody else holds back does not count either (it
    cannot start, so it must not hold up the lock owner's own launches; it is ahead again the moment the lock is gone). The tickets lie next to the GPU slots, so the
    problems that share the GPU share the line"""

    def __init__(self, base, run_id):
        self.dir, self.id, self.name = os.path.join(base, ".queue"), run_id, None

    def join(self):
        os.makedirs(self.dir, exist_ok=True)
        self.name = f"{time.time_ns():020d}_{uuid.uuid4().hex[:8]}"
        save_rec(os.path.join(self.dir, self.name), dict(id=self.id, pid=os.getpid(), since=now(), owner=os.environ.get("AR_LOCK_OWNER", "")), indent=None)

    def ahead(self):
        """the ids of the live tickets older than mine"""
        out = []
        try:
            names = sorted(os.listdir(self.dir))
        except OSError:
            return out
        for n in names:
            if n >= self.name:
                break
            f = os.path.join(self.dir, n)
            try:
                with open(f, encoding="utf-8") as fh:
                    o = json.load(fh)
                alive = pid_alive(o.get("pid"))
            except (OSError, ValueError, AttributeError):
                try:
                    alive = time.time() - os.path.getmtime(f) < 10      # being written right now, or a leftover
                except OSError:
                    continue
                o = {}
            if alive:
                if not blocked_by_lock(o.get("owner", "")):
                    out.append(o.get("id", "?"))
            else:
                try:
                    os.remove(f)
                except OSError:
                    pass
        return out

    def leave(self):
        if self.name:
            try:
                os.remove(os.path.join(self.dir, self.name))
            except OSError:
                pass
            self.name = None


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
    GpuSlot.stale); a torn read gave record() == {}, and a live slot with an unreadable owner file looked stale and was taken twice"""
    text = json.dumps(rec, indent=indent)
    for i in range(3):
        try:
            return P.atomic_write(path, text)
        except PermissionError:      # native Windows: os.replace over a file that a reader has open
            time.sleep(0.1)
    P.atomic_write(path, text)


def state_of(d):
    """done | ok | failed | stopped | held | blocked | gpu_busy | timeout | running | stalled | '' from a run directory. `held`: the run id has a HOLD marker
    (hold.py) and nothing runs: it does not start until the person's words clear it"""
    if os.path.exists(os.path.join(d, "done")):
        return "done"
    rec = record(d)
    st = rec.get("status", "")
    if st in ("running", "waiting"):
        if "launcher_pid" not in rec and not os.path.exists(os.path.join(d, "heartbeat")):
            return "running"  # a record of the launcher before heartbeats: nothing to judge it by
        return "stalled" if heartbeat_age(d) > STALE_S else "running"
    return "held" if HOLD.read(d) is not None else st


def beat(hb, text=""):
    open(hb, "w").write(f"{now()} {text}".rstrip() + "\n")


def ledger_row(ledger, run_id):
    """the ledger row of an id (empty dict when none)"""
    if not os.path.exists(ledger):
        return {}
    with open(ledger, newline="", encoding="utf-8") as f:
        return next((r for r in csv.DictReader(f) if r["id"] == run_id), {})


def _attempt_time(d):
    """when the attempt of a run record or ledger row began, to the second (spaces, microseconds and the like do not matter): its `started`, else its
    `ended` (a run that never started: blocked, gpu_busy); None when it has neither or cannot be read"""
    for k in ("started", "ended"):
        s = str(d.get(k) or "").strip()
        if s:
            try:
                return dt.datetime.strptime(s[:19].replace(" ", "T"), "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                return None
    return None


def later_attempt(rec, row):
    """is the attempt of the run record `rec` later than the one the ledger `row` describes? An unreadable or missing time on either side: no"""
    a, b = _attempt_time(rec), _attempt_time(row)
    return a is not None and b is not None and a > b


def ledger_write(ledger, row, clear=(), fill=False):
    """merge one row into the ledger under the lock (a planned row keeps what was written before the run; `clear` names columns of the old row to drop).
    fill=True (ledger.py sync): a missing row or a planned / running / stalled one is filled in; a finished row is replaced only by the row of a LATER
    attempt (later_attempt: the re-run whose own final write failed): only the card columns of the old row stay, its result and decision are kept as
    a note of the new row; the same attempt's row is never replaced; returns whether it wrote"""
    with P.locked(ledger):
        rows = []
        if os.path.exists(ledger):
            with open(ledger, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        old = next((r for r in rows if r["id"] == row["id"]), {})
        if fill and old.get("status") not in (None, "", "planned", "running", "stalled"):
            if not later_attempt(row, old):
                return False
            gone = "; ".join(f"{k} {old[k]}" for k in ("result", "decision") if old.get(k))
            old = {k: v for k, v in old.items() if k in ("id", "hypothesis", "basis", "seed", "prediction", "rule")}  # the card, not the earlier attempt's facts
            if gone:
                row = dict(row, notes=(row["notes"] + "; " if row.get("notes") else "") + f"replaced the row of the earlier attempt ({gone[:200]})")
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


def write_manifest(run_id):
    """the provenance manifest of the run (provenance.py manifest: GT, config, checkpoint, code tag, the model bundle), once at the end of every run of any status whose run
    folder exists (a training; an evaluation or a data job has none). Only a warning when it fails: it must never change the result or the exit code of the run"""
    try:
        import provenance
        if provenance.find_run_dir(run_id):
            provenance.manifest(run_id)
    except (Exception, SystemExit) as e:          # SystemExit: manifest() exits with a message when the run record is missing
        print(f"[{run_id}] warning: provenance manifest not written ({type(e).__name__}: {str(e)[:160]})", flush=True)


def hold_refusal(records, run_id):
    """'' when the run id may start, else why not: it has a HOLD marker, or another id has one and names the same run folder (trainsuite/compose.py: the
    folder is the id with . as _, so `t1.a` and `t1_a` share one folder and one checkpoint, but have two run records)"""
    folder = run_id.replace(".", "_")
    for name, h in HOLD.holds(records):
        if name.replace(".", "_") == folder:
            return (f"this run id is on HOLD ({HOLD.line(name, h)})" if name == run_id else
                    f"{name} is on HOLD and names the same run folder ({folder}) ({HOLD.line(name, h)})") + f". It starts again after the person's words: hold.py clear {name} --words \"<their words>\""
    return ""


def wait_after(records, ids, minutes, hb=None, stop=None):
    """wait for the done flags of ids; returns '' when all are done, else the reason; touches the heartbeat while waiting. `stop` = a function that returns
    a reason to give up (a hold set meanwhile) or ''"""
    t0 = time.time()
    while True:
        if hb:
            beat(hb, "waiting for " + ",".join(ids))
        why = stop() if stop else ""
        if why:
            return why
        missing = []
        for i in ids:
            st = state_of(os.path.join(records, i))
            if st in ("done", "ok"):  # ok = exit 0 whose done flag could not be written
                continue
            if st in ("failed", "gpu_busy", "blocked", "timeout", "stalled", "stopped", "held"):
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
    """one of n GPU slots: a directory taken with an atomic mkdir; a slot whose owner is dead, whose run is over, or that has no owner file is reclaimed
    (under a lock, so two launchers that judge the same slot stale do not both take it); the owner file carries a token and release() removes only
    the slot that still carries it"""

    def __init__(self, records, n, run_id, run_dir, base=None):
        self.base, self.n, self.id, self.run_dir, self.held = base or os.path.join(records, ".gpu_slots"), n, run_id, run_dir, None
        self.token = uuid.uuid4().hex

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

    def own(self, s):
        """write the owner file of slot s"""
        save_rec(os.path.join(s, "run.json"), dict(id=self.id, pid=os.getpid(), run_dir=self.run_dir, taken=now(), token=self.token), indent=None)
        self.held = s

    def reclaim(self, s):
        """take a slot that looked stale: serialised, the judgement is repeated inside the lock (a launcher that got there first has a live owner now)"""
        try:
            with P.locked(os.path.join(self.base, ".reclaim"), wait_s=10, stale_s=30):
                if not self.stale(s):
                    return False
                shutil.rmtree(s, ignore_errors=True)
                os.mkdir(s)
                self.own(s)
                return True
        except OSError:  # FileExistsError: a launcher took the freed slot with a plain mkdir; TimeoutError: the lock stays busy; both: next round
            return False

    def try_take(self):
        os.makedirs(self.base, exist_ok=True)
        for i in range(self.n):
            s = os.path.join(self.base, str(i))
            try:
                os.mkdir(s)
            except FileExistsError:
                if self.stale(s) and self.reclaim(s):
                    return True
                continue
            self.own(s)
            return True
        return False

    def settling(self, settle_s):
        """[(id, seconds since it took the slot)] of the other live launchers that took their slot less than settle_s ago. Their run has not loaded its data or allocated
        its GPU memory yet, so the RAM gate and the memory gate would read the box too early and start a second job next to it (two trainings that started in the same
        second on a 12 GB card slowed each other 13 times). A slot record without a time (an older launcher), a dead owner and a time in the future count for nothing"""
        out = []
        for i in range(self.n if settle_s > 0 else 0):
            s = os.path.join(self.base, str(i))
            o = record(s) if os.path.isdir(s) else {}
            if not o or o.get("token") == self.token or o.get("id") == self.id or not pid_alive(o.get("pid")):
                continue
            try:
                age = (dt.datetime.now() - dt.datetime.fromisoformat(o["taken"])).total_seconds()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= age < settle_s:
                out.append((o.get("id", "?"), int(age)))
        return out

    def release(self):
        if self.held and record(self.held).get("token") == self.token:  # a slot that was reclaimed belongs to its new owner
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


def run_once(cmd, cwd, log_path, hb_path, timeout_s, env=None):
    """one attempt: returns (exit code, text the attempt wrote, pid, killed by the timeout); the heartbeat is touched while the command runs"""
    start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    with open(log_path, "a", encoding="utf-8", errors="replace") as lg:
        lg.write(f"=== attempt {now()}: {' '.join(cmd)}\n")
        lg.flush()
        p = subprocess.Popen(cmd, cwd=cwd, stdout=lg, stderr=subprocess.STDOUT, start_new_session=POSIX, env=env)
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


SPLIT_ARG = re.compile(r"--split(?:\s+|\s*=\s*)[\"']?([\w,.-]+)", re.I)
SNAPSHOT_TRIES, SNAPSHOT_RETRY_S = 3, 2         # a snapshot that fails (a locked file on the Windows side, a busy index) is tried again before the run is refused


def names_split(cmd, split):
    """does the command line itself ask for this split (`--split test`, `--split=test`, `--split=val,test`, also inside `bash -c '... --split test'`)? The gate
    looks at --split of the launcher, which a command can contradict. Looked for in the joined command line, so a script string is read too. Best effort: the
    launcher's --split is the gate; an argparse prefix (`--spl test`), a split passed by environment variable or config file is not seen, and a quoted
    `--split test` inside an echo counts as a hit"""
    return any(split.lower() in m.group(1).lower().split(",") for m in SPLIT_ARG.finditer(" ".join(str(t) for t in cmd)))


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
    ap.add_argument("--settle-s", type=float, default=settle_s(cfg), help="GPU job: wait while another launcher took its slot less than this many seconds ago, so its data is loaded and its GPU memory allocated before the RAM / memory gates look (project.json gpu_settle_s; 0 = off)")
    ap.add_argument("--ram-mb", type=int, default=int(cfg.get("ram_mb", 0)), help="GPU job: start only when this much RAM is available (project.json ram_mb; 0 = no gate): two jobs run in parallel only if the RAM is there for both")
    ap.add_argument("--wait-gpu", type=int, default=720, help="minutes to wait for a GPU slot")
    ap.add_argument("--timeout", type=float, default=0, help="minutes after which the command is killed (0 = none)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--again", action="store_true", help="repeat an id that finished with exit 0 (otherwise refused)")
    ap.add_argument("--result", default="", help="json file the command writes its metrics to; embedded in the ledger when it exists")
    ap.add_argument("--no-git", action="store_true", help="do not snapshot and tag the code")
    ap.add_argument("--pin", metavar="REF", help="run on the code of this commit or tag (a resume that must continue on the code of attempt 1) instead of a snapshot of the working tree: "
                    "AR_COMMIT is that commit, no new commit or tag, no drift check; the record says so")
    ap.add_argument("--no-drift-check", action="store_true", help="do not look at the end of the run whether the code on disk changed since the snapshot")
    ap.add_argument("--cwd", default=root, help="working directory of the command (default: problem folder)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if cfg.get("gpu_slots_dir") and not os.path.isdir(os.path.dirname(slot_dir(root, cfg, records))):
        print(f"[{a.id}] warning: gpu_slots_dir {cfg['gpu_slots_dir']} lies in a folder that does not exist: the GPU slots are NOT shared with the other problem")
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
    holdout = str(cfg.get("holdout_split", "test")).lower()
    if a.split != holdout and names_split(cmd, holdout):
        print(f"[{a.id}] refused: the command asks for the holdout split ({holdout}) but --split is {a.split}; start it with --split {holdout} (the TEST log is checked then)")
        return 2
    if a.split == holdout:
        log = open(test_log, encoding="utf-8").read() if os.path.exists(test_log) else ""
        if not re.search(r"(?<![\w.])" + re.escape(a.id) + r"(?![\w.])", log):  # the whole id, not a prefix of a longer id
            print(f"[{a.id}] refused: split={a.split} is the holdout split but {a.id} is not named in {os.path.relpath(test_log, root)} (log it first, with the person's OK)")
            return 2
    if a.pin and a.no_git:
        print(f"[{a.id}] refused: --pin names a commit, --no-git has none")
        return 2
    if not a.no_git:
        err = git_identity_error(root)
        if err:
            print(f"[{a.id}] refused: git cannot commit the snapshot here ({err}); set user.name / user.email in the repository, or --no-git")
            return 2
    d = os.path.join(records, a.id)
    held = hold_refusal(records, a.id)
    if held:       # the person's decisions are acted on only when recorded (hard rule 6): a stopped run waits for their words, not for a chat's judgement
        print(f"[{a.id}] refused: {held}")
        return 2
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
        if a.pin and not prev.get("tag"):
            print(f"[{a.id}] refused: --pin continues an earlier attempt of this id (the tag of attempt 1); this id has none: start it without --pin")
            return 2
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

    def refuse_held():
        """a hold set while this launcher waits (--after, machine lock, GPU queue) is read again every round: the run does not start; status held, a ledger row, exit 2 like the first check"""
        why = hold_refusal(records, a.id)
        if why:
            rec.update(status="held", ended=now(), notes=why[:300])
            save_rec(os.path.join(d, "run.json"), rec)
            ledger_safe(ledger, dict(rec, exit=2, duration_s=0), clear)
            print(f"[{a.id}] refused: {why}")
        return why

    if a.after:
        why = wait_after(records, [x for x in a.after.split(",") if x], a.wait_after, hb, stop=lambda: hold_refusal(records, a.id))
        if refuse_held():
            return 2
        if why:
            rec.update(status="blocked", ended=now(), notes=why)
            save_rec(os.path.join(d, "run.json"), rec)
            ledger_safe(ledger, dict(rec, exit=3, duration_s=0), clear)
            print(f"[{a.id}] blocked: {why}")
            return 3
    slot = GpuSlot(records, a.gpu_slots, a.id, d, base=slot_dir(root, cfg, records)) if (a.gpu or a.gpu_max_mb) else None
    ticket = Ticket(slot.base, a.id) if slot else None           # first come, first served among the waiters for the GPU
    ram = RamGate(a.ram_mb) if (slot and a.ram_mb) else None
    owner = os.environ.get("AR_LOCK_OWNER", "")
    try:
        t0 = time.time()
        if ticket:
            ticket.join()
        while True:
            if refuse_held():               # the round that ends in the start (break) looks first, so the hold is read right before the snapshot too
                return 2
            try:
                lk = machine_lock.blocks(owner)  # D19: every run (GPU or not) waits while another owner holds an exclusive machine lock
            except Exception:  # noqa: BLE001  the lock must never be what breaks a launch
                lk = None
            if not lk and not slot:
                break
            ahead = ticket.ahead() if ticket else []
            loading = slot.settling(a.settle_s) if slot else []    # another launcher took its slot a moment ago: let it load before the gates look at the box
            u = gpu_used_mb() if (slot and a.gpu_max_mb) else None
            mem_wait = bool(a.gpu_max_mb) and not mem_ok(u, a.gpu_max_mb)
            beat(hb, f"waiting for the machine lock of {lk['owner']}" if lk else
                 f"waiting in the queue behind {', '.join(ahead[:3])}" if ahead else
                 f"waiting for {loading[0][0]} to load ({loading[0][1]} of {a.settle_s:g} s)" if loading else
                 f"waiting for GPU memory ({'unreadable' if u is None else str(u) + ' MB in use'}, the gate wants less than {a.gpu_max_mb} MB)" if mem_wait else
                 f"waiting for RAM ({ram.last} of {a.ram_mb} MB available)" if (ram and ram.n < RAM_CHECKS and ram.last is not None) else "waiting for the GPU")
            if not lk and not ahead and not loading and mem_ok(u, a.gpu_max_mb):
                if (ram is None or ram.passed()) and slot.try_take():
                    break
            elif ram:
                ram.reset()
            if time.time() - t0 > a.wait_gpu * 60:
                reasons = [f"{len(ahead)} run(s) ahead in the queue ({', '.join(ahead[:3])})"] if ahead else []
                if loading:
                    reasons.append(f"{loading[0][0]} took its slot {loading[0][1]} s ago and is still loading ({a.settle_s:g} s settle time)")
                if mem_wait:
                    reasons.append("nvidia-smi unreadable" if u is None else f"GPU memory {u} MB in use (the gate wants less than {a.gpu_max_mb} MB)")
                if ram and ram.last is not None and ram.n < RAM_CHECKS:
                    reasons.append(f"RAM {ram.last} of {a.ram_mb} MB available")
                taken = sum(os.path.isdir(os.path.join(slot.base, str(i))) for i in range(a.gpu_slots))
                why = (f"machine lock of {lk['owner']} until {lk['end']}" if lk else ", ".join(reasons + [f"{taken} of {a.gpu_slots} GPU slots taken"]))
                rec.update(status="gpu_busy", ended=now(), notes=why + f" for {a.wait_gpu} min")
                save_rec(os.path.join(d, "run.json"), rec)
                ledger_safe(ledger, dict(rec, exit=1, duration_s=0), clear)
                print("machine lock not released:" if lk else "GPU did not become free:", rec["notes"])
                return 1
            time.sleep(min(30, max(1, HEARTBEAT_S)))
        if ticket:
            ticket.leave()                                       # the place in the line is given up as soon as the run starts, not when it ends
        ch = code_hash(root, cfg)
        if prev.get("code_hash") and prev["code_hash"] != ch and not a.pin:
            notes.append(f"code changed since attempt 1 ({prev['code_hash']} -> {ch})")
        try:
            before = drift_hash(root, cfg)  # before the snapshot: a change while the snapshot is made counts, the precise check at the end tells
        except OSError:  # a scratch file that vanished during the walk must not stop the launch: no hash, no drift check
            before = None
        for tries in range(1, SNAPSHOT_TRIES + 1):
            if a.pin:
                commit, tag, gnote = pin_snapshot(root, a.pin, prev)
            else:
                commit, tag, gnote = ("skipped", "", "") if a.no_git else git_snapshot(root, cfg, cfg["tag_prefix"], a.id, a.hyp or " ".join(cmd))
            if commit != "git-error" or tries == SNAPSHOT_TRIES:
                break
            time.sleep(SNAPSHOT_RETRY_S)
        if commit == "git-error":        # a run must name the code it ran: no snapshot, no start (--no-git is the explicit way to run without one)
            rec.update(status="failed", ended=now(), notes=f"the code snapshot failed {tries} times, the command was not started: {gnote}"[:300])
            save_rec(os.path.join(d, "run.json"), rec)
            ledger_safe(ledger, dict(rec, exit=1, duration_s=0), clear)
            print(f"[{a.id}] refused: the code snapshot failed ({gnote}); fix git here, or --no-git to run without a snapshot")
            return 1
        if gnote:
            notes.append(gnote)
        rec.update(status="running", started=now(), code_hash=(prev.get("code_hash") or ch) if a.pin else ch, commit=commit, tag=tag or prev.get("tag", ""), git_note=gnote)
        save_rec(os.path.join(d, "run.json"), rec)
        t1 = time.time()
        rc, attempt, tail, timed_out = 1, 0, "", False
        # the command learns which code snapshot it is (a launcher that trains from `git archive $AR_COMMIT` runs exactly the recorded code,
        # also on a resumed id whose tag still names attempt 1; any other command runs from the working tree: see code_drift below)
        snapshotted = commit not in ("skipped", "no-git", "git-error")
        try:
            launch_spec = snapshot_spec(root, cfg)[1] if snapshotted and not a.no_drift_check and not a.pin else None    # a pinned run runs another commit than the disk: nothing to compare
        except Exception:
            launch_spec = None
        cmd_env = dict(os.environ, AR_RUN_ID=a.id, **({"AR_COMMIT": commit, "AR_TAG": tag or prev.get("tag", "")} if snapshotted else {}))
        while attempt <= a.retries:
            attempt += 1
            try:
                rc, out, pid, timed_out = run_once(cmd, a.cwd, os.path.join(d, "run.log"), hb, a.timeout * 60, cmd_env)
            except Exception as e:  # the command could not even be started (not found, bad cwd): a failed run, not a dead launcher
                rc, pid = 127, None
                out = f"launcher could not start the command: {type(e).__name__}: {e}"
                with open(os.path.join(d, "run.log"), "a", encoding="utf-8", errors="replace") as lg:
                    lg.write(out + "\n")
            rec["pid"] = pid
            tail = out[-3000:]
            if rc == 0 or timed_out or rc == EXIT_STOPPED or not CUDA_RETRY.search(out):   # a stop is final even when the log tail holds an older CUDA error
                break
            if attempt <= a.retries:
                time.sleep(15)
        dur = round(time.time() - t1)
    finally:
        if ticket:
            ticket.leave()
        if slot:
            slot.release()
    res = ""
    if a.result and os.path.exists(a.result):
        try:
            res = json.dumps(json.load(open(a.result)))[:400]
        except Exception:
            res = "(result file unreadable)"
    if snapshotted and not a.no_drift_check and not a.pin:  # the commit names the code at launch; did the code on disk change while the command ran? (not for --pin: it ran another commit)
        drift, drift_err = code_drift(root, cfg, commit, before, launch_spec)
        if drift:
            rec["code_drift"] = drift
            notes.append(f"code on disk changed while the run was going ({drift['n']} files, e.g. {', '.join(drift['paths'][:3])}): the commit/tag names the code at launch, "
                         "which a command that does not execute AR_COMMIT may not have run")
        if drift_err:
            rec["code_drift_check"] = drift_err
    status = "ok" if rc == 0 else "timeout" if timed_out else "stopped" if rc == EXIT_STOPPED else "failed"  # a command's own exit 124 is a failure, not the launcher's timeout
    if status == "stopped":
        HOLD.write(d, "run.py", "exit 75: the trainer was stopped on request (trainsuite), a checkpoint should have been saved", "(none recorded: ask the person; the stop request is in the run folder's STOPPED file)")
        notes.append("stopped on request (exit 75 = the trainsuite stop code, a checkpoint should have been saved) and ON HOLD: the same id starts again only after the person's words "
                     "(hold.py clear); its dependents are blocked meanwhile")
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
    write_manifest(a.id)
    print(f"[{a.id}] {status} (exit {rc}, {dur} s, {attempt} attempt(s)), code {commit} {rec['tag']}; log {os.path.join(d, 'run.log')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
