"""Find the problem folder and read its project.json (standard library only; imported by every tool of autoresearch).

The problem folder comes from the environment variable AR_PROJECT, else from autoresearch/config.json (default_project, relative to
autoresearch/). Paths in project.json are relative to the problem folder; P.path("ledger") returns an absolute path.
"""
import contextlib
import json
import os
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.dirname(HERE)

DEFAULTS = {
    "tag_prefix": "exp",
    "ledger": "LEDGER.csv",
    "ledger_md": "LEDGER.md",
    "run_records": "runs_records",
    "schedule": "schedule.csv",
    "test_log": "TEST_USAGE_LOG.md",
    "code_dirs": [],
    "fingerprint_extra": [],
    "scratch": "temp",
    "live_code": [],
    "trash": "_to_delete",
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


def _holds(lock, token):
    """does the lock file still carry this holder's token (last field of '<pid> <time> <token>')? A lock that was taken over is not ours any more"""
    try:
        with open(lock) as f:
            return f.read().split()[-1:] == [token]
    except OSError:
        return False


@contextlib.contextmanager
def locked(path, wait_s=None, stale_s=120):
    """exclusive lock on a shared state file (ledger, schedule): <path>.lock is created atomically with '<pid> <time> <token>' and its mtime is renewed every
    stale_s / 4 s while the holder is inside the block, so only a dead or frozen holder ages out: a lock not renewed for stale_s is taken over. The holder
    removes the lock only when the file still carries its token"""
    lock = path + ".lock"
    wait_s = float(os.environ.get("AR_LOCK_WAIT_S", "60")) if wait_s is None else wait_s
    token = uuid.uuid4().hex
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()} {token}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    gone = f"{lock}.stale.{os.getpid()}"
                    os.rename(lock, gone)  # only one waiter wins the rename; the others see FileNotFoundError and retry
                    if time.time() - os.path.getmtime(gone) <= stale_s and not os.path.exists(lock):
                        os.rename(gone, lock)  # not stale after all (another waiter's takeover came first, and this is its fresh lock): put it back and wait
                    else:
                        os.remove(gone)
                        continue
            except OSError:
                continue
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
    beat.start()
    try:
        yield
    finally:
        done.set()
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
