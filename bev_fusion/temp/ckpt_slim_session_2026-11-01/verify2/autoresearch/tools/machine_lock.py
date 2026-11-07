"""Machine lock between chats (D19): one lock file, autoresearch/state/machine_lock.json (not in git), that every launcher honours. Standard
library only.

    python3 machine_lock.py take <owner> --hours 2 [--exclusive] [--idle] [--note "E253 timing"]   refused while another owner holds a live lock;
            --idle: refused unless the GPU is idle (utilisation <= 5 %, no compute process), the minimum of D19 (c) for a timing driver
    python3 machine_lock.py release <owner>          only the owner releases (an expired lock needs no release)
    python3 machine_lock.py show                     the lock, or "free"
    python3 machine_lock.py wait <owner> [--minutes 720]   for shell launchers: returns (exit 0) once no other owner holds a live exclusive lock,
            exit 1 on timeout
    python3 machine_lock.py --selftest

A lock is live until its `end`. Exclusive = nobody else starts a run: `run.py` waits before every run (GPU or not) while another owner holds a live
exclusive lock (its owner is AR_LOCK_OWNER of the launcher's environment, so the lock owner's own runs pass). A non-exclusive lock only announces a
reservation (`show`). The file path can be moved with AR_MACHINE_LOCK (selftest).
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402


def lock_path():
    """AR_MACHINE_LOCK, else autoresearch/state/ of the MAIN checkout (a worktree's run.py honours the same lock), else of this checkout"""
    if os.environ.get("AR_MACHINE_LOCK"):
        return os.environ["AR_MACHINE_LOCK"]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        common = subprocess.run(["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        common = ""
    main = os.path.join(os.path.dirname(common), "autoresearch") if common.endswith(".git") else ""
    return os.path.join(main if main and os.path.isdir(main) else here, "state", "machine_lock.json")


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def current():
    """the live lock as a dict, or None (no file, unreadable, or past its end)"""
    try:
        lk = json.load(open(lock_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = isinstance(lk, dict) and isinstance(lk.get("end"), str) and isinstance(lk.get("owner"), str)  # a malformed file never blocks the machine
    return lk if ok and lk["end"] > now() else None


def blocks(owner):
    """the live exclusive lock of another owner that stops a run of `owner`, or None"""
    lk = current()
    return lk if lk and lk.get("exclusive") is True and lk["owner"] != owner else None


def gpu_idle():
    """(idle, why): utilisation <= 5 % and no compute process; a GPU that cannot be read is not idle"""
    try:
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        u = int(util.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return False, "nvidia-smi unreadable"
    n = len([x for x in apps.split() if x.strip()])
    return (u <= 5 and n == 0), f"GPU {u} %, {n} compute process(es)"


def take(owner, hours, exclusive=False, idle=False, note="", idle_fn=gpu_idle):
    p = lock_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with P.locked(p):
        lk = current()
        if lk and lk.get("owner") != owner:
            sys.exit(f"machine lock held by {lk['owner']} until {lk['end']} ({'exclusive' if lk.get('exclusive') else 'reservation'}: {lk.get('note', '')})")
        if idle:
            ok, why = idle_fn()
            if not ok:
                sys.exit(f"not idle: {why}")
        row = dict(owner=owner, start=now(), end=(dt.datetime.now() + dt.timedelta(hours=hours)).isoformat(timespec="seconds"), exclusive=bool(exclusive), note=note)
        P.atomic_write(p, json.dumps(row, indent=1) + "\n")
    print(f"machine lock: {owner} until {row['end']} ({'exclusive' if exclusive else 'reservation'})")
    return row


def release(owner):
    p = lock_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with P.locked(p):
        lk = current()
        if lk and lk.get("owner") != owner:
            sys.exit(f"machine lock belongs to {lk['owner']}, not {owner}")
        if os.path.exists(p):
            os.replace(p, p + ".released")  # the last lock stays readable for a post-mortem
    print("machine lock: free")


def wait(owner, minutes, poll_s=30):
    t0 = time.time()
    while True:
        lk = blocks(owner)
        if not lk:
            return 0
        if time.time() - t0 > minutes * 60:
            print(f"machine lock of {lk['owner']} until {lk['end']} still held after {minutes} min")
            return 1
        time.sleep(poll_s)


def selftest():
    import shutil
    import tempfile
    d, old = tempfile.mkdtemp(), os.environ.get("AR_MACHINE_LOCK")
    os.environ["AR_MACHINE_LOCK"] = os.path.join(d, "state", "machine_lock.json")
    assert current() is None and blocks("a") is None, "no file: free"
    release("nobody")  # no state folder yet: nothing to release, no crash
    take("timing", 1, exclusive=True, note="t")
    assert blocks("bev") and not blocks("timing"), "an exclusive lock stops others, not its owner"
    for bad in (lambda: take("bev", 1), lambda: release("bev")):
        try:
            bad()
            raise AssertionError("another owner can neither take nor release a live lock")
        except SystemExit:
            pass
    assert wait("bev", 0, poll_s=0) == 1 and wait("timing", 0) == 0
    release("timing")
    assert current() is None
    take("bev", 1, exclusive=False)
    assert blocks("timing") is None, "a reservation stops nobody"
    json.dump(dict(owner="old", end="2000-01-01T00:00:00", exclusive=True), open(os.environ["AR_MACHINE_LOCK"], "w"))
    assert current() is None and blocks("x") is None, "an expired lock is free"
    take("x", 1)  # over an expired lock
    release("x")
    try:
        take("t2", 1, exclusive=True, idle=True, idle_fn=lambda: (False, "GPU 40 %, 1 compute process(es)"))
        raise AssertionError("--idle refuses a busy GPU")
    except SystemExit:
        pass
    assert current() is None, "a refused take leaves no lock"
    open(os.environ["AR_MACHINE_LOCK"], "w").write("{torn")
    assert current() is None, "an unreadable file is free (a crash never blocks the machine)"
    for bad in (dict(owner="x", end=None, exclusive=True), dict(owner="x", end=5, exclusive=True), dict(end="2999-01-01", exclusive=True), [1]):
        json.dump(bad, open(os.environ["AR_MACHINE_LOCK"], "w"))
        assert current() is None and blocks("y") is None, f"a malformed lock is free: {bad}"
    json.dump(dict(owner="x", end="2999-01-01", exclusive="false"), open(os.environ["AR_MACHINE_LOCK"], "w"))
    assert blocks("y") is None, "only exclusive: true is exclusive"
    shutil.rmtree(d)
    if old is None:
        del os.environ["AR_MACHINE_LOCK"]
    else:
        os.environ["AR_MACHINE_LOCK"] = old
    print("machine_lock selftest passed")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["take", "release", "show", "wait"])
    ap.add_argument("owner", nargs="?", default="")
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--minutes", type=float, default=720)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--idle", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd in ("take", "release", "wait") and not a.owner:
        sys.exit(f"{a.cmd} needs an owner")
    if a.cmd == "take":
        take(a.owner, a.hours, a.exclusive, a.idle, a.note)
    elif a.cmd == "release":
        release(a.owner)
    elif a.cmd == "wait":
        sys.exit(wait(a.owner, a.minutes))
    else:
        lk = current()
        print(json.dumps(lk, indent=1) if lk else "free")


if __name__ == "__main__":
    main()
