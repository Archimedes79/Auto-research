"""Find the problem folder and read its project.json (standard library only; imported by every tool of autoresearch).

The problem folder comes from the environment variable AR_PROJECT, else from autoresearch/config.json (default_project, relative to
autoresearch/). Paths in project.json are relative to the problem folder; P.path("ledger") returns an absolute path.
"""
import contextlib
import csv
import json
import os
import socket
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.dirname(HERE)

try:        # the csv module stops at 131072 characters per field: two long `ledger.py note --notes` made every later read of the ledger fail. Every tool imports this module
    csv.field_size_limit(sys.maxsize)
except OverflowError:   # a C long of 32 bit
    csv.field_size_limit(2 ** 31 - 1)

DEFAULTS = {
    "tag_prefix": "exp",
    "ledger": "LEDGER.csv",
    "ledger_md": "LEDGER.md",
    "run_records": "run_records",
    "runs": "runs",     # the parent of the folders a run writes into (<runs>/<id>, or the id with . as _): provenance, package. NOT "cards": that is where the evaluator writes its cards
    "schedule": "schedule.csv",
    "test_log": "TEST_USAGE_LOG.md",
    "code_dirs": [],
    "fingerprint_extra": [],
    "scratch": "temp",
    "live_code": [],
    "trash": "_to_delete",
    "experiment_dirs": ["experiments"],
    "structure": {},
}


def root():
    p = os.environ.get("AR_PROJECT")
    if not p:
        p = json.load(open(os.path.join(AR, "config.json"), encoding="utf-8"))["default_project"]
    r = os.path.normpath(p if os.path.isabs(p) else os.path.join(AR, p))
    if not os.path.isfile(os.path.join(r, "project.json")) and not os.path.isabs(p):
        alt = os.path.normpath(os.path.abspath(p))              # a relative AR_PROJECT may also mean: relative to the current folder
        if os.path.isfile(os.path.join(alt, "project.json")):
            r = alt
            os.environ["AR_PROJECT"] = r                        # a child started from another folder must find the same problem
    if not os.path.isfile(os.path.join(r, "project.json")):    # defaults + a ghost ledger folder would follow otherwise
        sys.exit(f"no project.json in {r} (AR_PROJECT={os.environ.get('AR_PROJECT')!r}, else default_project of autoresearch/config.json): "
                 "that folder is not a problem folder")
    return r


def cfg_of(problem_root):
    """the project.json of a given problem folder, with the defaults (cfg() is the same for the default / AR_PROJECT problem)"""
    c = dict(DEFAULTS)
    f = os.path.join(problem_root, "project.json")
    if os.path.exists(f):
        c.update({k: v for k, v in json.load(open(f, encoding="utf-8")).items() if not k.startswith("_")})
    return c


def cfg():
    c = dict(DEFAULTS)
    f = os.path.join(root(), "project.json")
    if os.path.exists(f):
        c.update({k: v for k, v in json.load(open(f, encoding="utf-8")).items() if not k.startswith("_")})
    return c


def path(key):
    return os.path.join(root(), cfg()[key])


def experiment_dirs(problem_root=None):
    """absolute paths of the folders that hold experiment folders: project.json "experiment_dirs" (default ["experiments"]; bev_fusion adds
    results/experiments for the finished ones, so an open card and a finished one are both found by id)"""
    r = problem_root or root()
    return [os.path.join(r, d) for d in cfg_of(r)["experiment_dirs"]]


_WHO = f"{sys.platform}:{socket.gethostname()}".replace(" ", "_")  # the machine and operating system a pid belongs to (WSL and Windows number their processes apart)


def _holder_dead(lock):
    """True when the lock names a holder of THIS machine and operating system ('<pid> <time> <who> <token>') whose process no longer runs: waiters take such a lock over
    at once instead of after stale_s. A lock of the older three-field form, of another machine or system, or an unreadable one is never judged dead"""
    try:
        with open(lock, encoding="utf-8") as f:
            parts = f.read().split()
        if len(parts) != 4 or parts[2] != _WHO or os.name == "nt":
            return False
        os.kill(int(parts[0]), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False


def _holds(lock, token):
    """does the lock file still carry this holder's token (last field of '<pid> <time> <token>')? A lock that was taken over is not ours any more"""
    try:
        with open(lock) as f:
            return f.read().split()[-1:] == [token]
    except OSError:
        return False


EMPTY_S = 10    # a lock file is empty only for the milliseconds between its creation and its write: an empty or unreadable one this old has no holder
GUARD_S = 5     # the takeover guard <lock>.takeover exists for microseconds: older than this (a taker killed inside it) it is removed, not waited for stale_s


def _has_text(lock):
    """does the lock file hold readable text? (empty: its creator died before the write; unreadable: nobody can say who holds it)"""
    try:
        with open(lock, encoding="utf-8") as f:
            return bool(f.read().split())
    except (OSError, ValueError):
        return False


def _stale(lock, stale_s):
    """is the lock file there and one that nobody holds any more: not renewed for stale_s, its holder (of this machine and system) dead, or no readable text in it
    for EMPTY_S. A file that is gone is not stale (there is nothing to remove, and whoever removes a lock that appears meanwhile removes a live holder's)"""
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return False
    return age > stale_s or _holder_dead(lock) or (age > EMPTY_S and not _has_text(lock))


def _take_over(lock, stale_s):
    """remove the lock of a holder that is gone; True when the lock is gone (removed here or vanished: create it now), False when it is alive or another waiter is
    taking it over. The taker holds a guard file <lock>.takeover (created with O_EXCL), judges the lock AGAIN under it and drops the guard at once: a waiter that judged
    the old lock stale a moment ago can no longer remove the fresh lock of the waiter that took over before it; a lock that vanished meanwhile is never removed. A guard
    older than GUARD_S is itself stale (its taker died)"""
    if not os.path.exists(lock):
        return True
    if not _stale(lock, stale_s):
        return False
    guard = lock + ".takeover"
    try:
        os.close(os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(guard) > GUARD_S:        # the guard is held for microseconds: one this old belongs to a taker that died
                os.remove(guard)
        except OSError:
            pass
        return False
    try:
        if not os.path.exists(lock):
            return True
        if not _stale(lock, stale_s):
            return False
        os.remove(lock)
        return True
    except OSError:
        return False
    finally:
        try:
            os.remove(guard)
        except OSError:
            pass


@contextlib.contextmanager
def locked(path, wait_s=None, stale_s=120):
    """exclusive lock on a shared state file (ledger, schedule): <path>.lock is created atomically with '<pid> <time> <who> <token>' and its mtime is renewed every
    stale_s / 4 s while the holder is inside the block, so only a dead or frozen holder ages out: a lock not renewed for stale_s, one of a dead holder of this machine,
    or an empty one older than EMPTY_S is taken over (_take_over: one waiter at a time, under a guard file). The holder removes the lock only when the file still
    carries its token. Known residual: a frozen holder (not renewed for stale_s) that wakes up exactly while a waiter removes its lock. The folder of the lock is made"""
    lock = os.path.abspath(path) + ".lock"
    os.makedirs(os.path.dirname(lock), exist_ok=True)         # a first `ledger.py add` / `schedule.py add` in a problem without its records folder
    wait_s = float(os.environ.get("AR_LOCK_WAIT_S", "60")) if wait_s is None else wait_s
    token = uuid.uuid4().hex
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()} {_WHO} {token}".encode())
            os.close(fd)
            break
        except FileExistsError:
            if _take_over(lock, stale_s):
                continue                                       # removed: try to create it at once
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"{lock} held for more than {wait_s} s")
            time.sleep(0.2)
    done = threading.Event()

    def renew():
        while not done.wait(max(0.05, stale_s / 4)):
            if _holds(lock, token):
                try:
                    os.utime(lock)
                except OSError:
                    pass
    beat = threading.Thread(target=renew, daemon=True)
    try:
        beat.start()
        yield
    finally:
        done.set()
        if beat.is_alive():
            beat.join()
        if _holds(lock, token):
            try:
                os.remove(lock)
            except OSError:
                pass


def atomic_write(path, text):
    """write the whole file through a temporary file and an atomic replace (a crash mid-write never leaves a truncated record)"""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
