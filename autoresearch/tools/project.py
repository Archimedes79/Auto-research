"""Find the problem folder and read its project.json (standard library only; imported by every tool of autoresearch).

The problem folder comes from the environment variable AR_PROJECT, else from autoresearch/config.json (default_project, relative to
autoresearch/). Paths in project.json are relative to the problem folder; P.path("ledger") returns an absolute path.
"""
import contextlib
import json
import os
import time

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
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(AR, p))


def cfg():
    c = dict(DEFAULTS)
    f = os.path.join(root(), "project.json")
    if os.path.exists(f):
        c.update({k: v for k, v in json.load(open(f, encoding="utf-8")).items() if not k.startswith("_")})
    return c


def path(key):
    return os.path.join(root(), cfg()[key])


@contextlib.contextmanager
def locked(path, wait_s=60, stale_s=120):
    """exclusive lock on a shared state file (ledger, schedule): <path>.lock is created atomically; a lock older than stale_s is taken over"""
    lock = path + ".lock"
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    os.remove(lock)
                    continue
            except OSError:
                continue
            if time.time() - t0 > wait_s:
                raise TimeoutError(f"{lock} held for more than {wait_s} s")
            time.sleep(0.2)
    try:
        yield
    finally:
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
