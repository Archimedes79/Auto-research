"""Scratch hygiene for bev_fusion/temp: the rules that keep it small (C: was 100 % full with 18 GB in temp/, 2026-11-01).

    with tmpdir("smoke_x") as d: ...      a fresh folder under temp/scratch/, removed on exit (also after an error; BF_KEEP_SCRATCH=1 keeps it
                                          for debugging). Smoke runs and test outputs go here, so nothing is left behind.
    cached(kind, build, n, deps, split)   a list cache of the FIRST n items of an ordered build: a cache with n' >= n is reused (sliced), a cache
                                          older than one of its inputs (deps) is deleted, and a smaller cache is deleted once a bigger one exists
    dump(kind, key, shape, dtype)         an .npy dump keyed by what it holds: a complete one is reused and never written twice; a new one is
                                          checked by guard() first (no double probing)
    guard(nbytes)                         refuses a big write that would leave less than 15 GB free on the drive or pass the 10 GB budget of
                                          temp/ (BF_SCRATCH_BUDGET_GB, BF_SCRATCH_KEEP_FREE_GB)

    python training/scratch.py report     temp/ by folder, extension and age, the budget and the free space
    python training/scratch.py --selftest
The finished checkpoints of a training run are slimmed by train.py itself (TrainConfig.slim_at_end), see training/README.md.
"""
import contextlib
import glob
import os
import re
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the bev_fusion folder
TEMP = os.path.join(ROOT, "temp")


class ScratchError(RuntimeError):
    pass


def size_gb(path):
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / 1e9


def guard(nbytes, keep_free_gb=None, budget_gb=None):
    """raise ScratchError if writing `nbytes` would leave less than keep_free_gb free on the drive of temp/ or push temp/ over budget_gb"""
    keep = float(os.environ.get("BF_SCRATCH_KEEP_FREE_GB", 15.0)) if keep_free_gb is None else keep_free_gb
    budget = float(os.environ.get("BF_SCRATCH_BUDGET_GB", 10.0)) if budget_gb is None else budget_gb
    gb = nbytes / 1e9
    free = shutil.disk_usage(ROOT).free / 1e9
    if free - gb < keep:
        raise ScratchError(f"writing {gb:.1f} GB would leave {free - gb:.1f} GB free (limit {keep:.0f} GB): clean up or move scratch first "
                           f"(python training/scratch.py report)")
    used = size_gb(TEMP) if os.path.isdir(TEMP) else 0.0
    if used + gb > budget:
        raise ScratchError(f"temp/ holds {used:.1f} GB, +{gb:.1f} GB passes the budget of {budget:.0f} GB: clean up first "
                           f"(python training/scratch.py report)")


@contextlib.contextmanager
def tmpdir(name="run"):
    base = os.path.join(TEMP, "scratch")
    os.makedirs(base, exist_ok=True)
    d = tempfile.mkdtemp(prefix=name + "_", dir=base)
    try:
        yield d
    finally:
        if os.environ.get("BF_KEEP_SCRATCH") == "1":
            print(f"scratch kept: {d}", flush=True)
        else:
            shutil.rmtree(d, ignore_errors=True)


def newest_mtime(deps):
    """mtime of the newest of deps: a file, a FOLDER (its own time: it changes when a file is added, removed or replaced by rename, and costs
    one stat where a glob over 20 000 files on /mnt/c takes minutes) or a glob pattern; 0 for none"""
    newest = 0.0
    for dep in deps:
        for p in (glob.glob(dep) if any(c in dep for c in "*?[") else [dep]):
            if os.path.exists(p):
                newest = max(newest, os.path.getmtime(p))
    return newest


def source_sig(*funcs):
    """short hash of the source text of the functions that build a cache: editing one makes the caches built by the old text outdated"""
    import hashlib
    import inspect
    h = hashlib.md5()
    for f in funcs:
        h.update(inspect.getsource(f).encode())
    return h.hexdigest()[:10]


def _meta(path, write=None):
    import json
    p = path + ".meta"
    if write is not None:
        with open(p, "w") as f:
            json.dump(write, f)
    return json.load(open(p)) if os.path.exists(p) else None


def _remove(path):
    for p in (path, path + ".meta"):
        if os.path.exists(p):
            os.remove(p)


def cached(kind, build, n, deps=(), split="", root=None, sig=None):
    """the first n items of an ordered build(n) -> list, cached as <root>/<kind>_<split>_<n>.pt (root default temp/checks) with a sidecar
    <cache>.meta {sig, deps}. Reuses a cache with n' >= n (sliced). A cache is OUTDATED, and deleted, if one of `deps` (a file, a folder or a
    glob) changed after it was built or the builder's `sig` (source_sig) differs. A cache without a sidecar (made before this rule) is
    adopted as valid as of now. After writing a new cache the smaller ones, which it contains, are deleted."""
    import torch
    d = root or os.path.join(TEMP, "checks")
    os.makedirs(d, exist_ok=True)
    stem = f"{kind}_{split}_" if split else f"{kind}_"
    found = {}
    for f in os.listdir(d):
        m = re.fullmatch(re.escape(stem) + r"(\d+)\.pt", f)
        if m:
            found[int(m.group(1))] = os.path.join(d, f)
    newest = newest_mtime(deps)
    for k, p in list(found.items()):
        meta = _meta(p)
        if meta is None:
            _meta(p, {"sig": sig, "deps": newest})                    # adopted: valid as far as we know, checked against changes from now on
        elif newest > meta.get("deps", 0) or (sig is not None and meta.get("sig") not in (None, sig)):
            _remove(p)                                                # outdated: an input or the builder changed after it was made
            del found[k]
            print(f"scratch: deleted the outdated cache {os.path.basename(p)}", flush=True)
    bigger = [k for k in found if k >= n]
    if bigger:
        obj = torch.load(found[min(bigger)], weights_only=False)      # our own scratch cache
        return obj[:n]
    obj = build(n)
    path = os.path.join(d, f"{stem}{n}.pt")
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    _meta(path, {"sig": sig, "deps": newest})                         # the deps' state BEFORE the build: a change during it shows next time
    for p in found.values():                                          # superseded: a smaller cache is a prefix of this one
        _remove(p)
        print(f"scratch: deleted {os.path.basename(p)} (contained in {os.path.basename(path)})", flush=True)
    return obj


def dump(kind, key, shape, dtype, root=None):
    """-> (array, path, complete). <root or temp/dumps/kind>/<key>.npy as a memmap. complete=True: the dump was finished before (mark_done),
    reuse it, do not fill it again. A new or unfinished dump is opened writable (new ones pass guard() first). A file of another shape or
    dtype under the same key raises: the key must name what the dump holds."""
    import numpy as np
    dtype, shape = np.dtype(dtype), tuple(int(s) for s in shape)
    p = os.path.join(root or os.path.join(TEMP, "dumps", kind), f"{key}.npy")
    if os.path.exists(p):
        a = np.load(p, mmap_mode="r")
        if a.shape != shape or a.dtype != dtype:
            raise ScratchError(f"{p} holds {a.shape} {a.dtype}, wanted {shape} {dtype}: delete it or use another key")
        done = os.path.exists(p + ".done")
        return (a if done else np.load(p, mmap_mode="r+")), p, done
    guard(int(np.prod(shape)) * dtype.itemsize)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return np.lib.format.open_memmap(p, "w+", dtype, shape), p, False


def mark_done(path):
    with open(path + ".done", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def report():
    if not os.path.isdir(TEMP):
        print("no temp folder")
        return
    now = time.time()
    rows, ext = [], {}
    for name in sorted(os.listdir(TEMP)):
        p = os.path.join(TEMP, name)
        if not os.path.isdir(p):
            continue
        newest, total = 0.0, 0
        for dp, _, fs in os.walk(p):
            for f in fs:
                fp = os.path.join(dp, f)
                try:
                    s = os.stat(fp)
                except OSError:
                    continue
                total += s.st_size
                newest = max(newest, s.st_mtime)
                e = os.path.splitext(f)[1] or "(none)"
                ext[e] = ext.get(e, 0) + s.st_size
        rows.append((total, name, (now - newest) / 86400 if newest else float("nan")))
    for total, name, age in sorted(rows, reverse=True):
        print(f"{total / 1e9:8.2f} GB  {name:<24} newest file {age:5.1f} days old")
    print("by extension: " + ", ".join(f"{e} {s / 1e9:.2f} GB" for e, s in sorted(ext.items(), key=lambda x: -x[1])[:6]))
    used, budget = sum(r[0] for r in rows) / 1e9, float(os.environ.get("BF_SCRATCH_BUDGET_GB", 10.0))
    print(f"temp/ {used:.1f} GB of the {budget:.0f} GB budget; {shutil.disk_usage(ROOT).free / 1e9:.1f} GB free on the drive")


def selftest():
    import numpy as np
    import torch
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print("%s  %s" % ("PASS" if cond else "FAIL", name))

    with tmpdir("selftest") as d:
        inside = os.path.isdir(d) and os.path.dirname(d) == os.path.join(TEMP, "scratch")
    check("tmpdir: made under temp/scratch, removed on exit", inside and not os.path.exists(d))
    try:
        with tmpdir("selftest") as d2:
            raise ValueError("planted")
    except ValueError:
        pass
    check("tmpdir: removed also after an error", not os.path.exists(d2))

    real = sorted(glob.glob(os.path.join(ROOT, "data", "cache", "gt", "*.npz")))[:600]       # real files, real names
    with tmpdir("selftest_cache") as d:
        calls = []

        def build(m):
            calls.append(m)
            return [os.path.basename(p) for p in real[:m]]

        dep = os.path.join(d, "input.txt")
        open(dep, "w").write("x")
        a = cached("k", build, 300, deps=(dep,), split="train", root=d)
        b = cached("k", build, 100, deps=(dep,), split="train", root=d)
        check("cached: a bigger cache is reused for a smaller n (sliced, no rebuild)", calls == [300] and b == a[:100] and len(a) == 300)
        c = cached("k", build, 500, deps=(dep,), split="train", root=d)
        left = sorted(os.listdir(d))
        check("cached: a bigger build deletes the smaller cache (and its sidecar)",
              calls == [300, 500] and left == ["input.txt", "k_train_500.pt", "k_train_500.pt.meta"] and c[:300] == a)
        t = time.time() + 5
        os.utime(dep, (t, t))
        cached("k", build, 500, deps=(dep,), split="train", root=d)
        check("cached: an input newer than the cache deletes it and rebuilds", calls == [300, 500, 500])
        cached("k", build, 50, deps=(dep,), split="test", root=d)
        check("cached: split 'test' is separate from 'train'", sorted(f for f in os.listdir(d) if f.endswith(".pt")) == ["k_test_50.pt", "k_train_500.pt"])
        n0 = len(calls)
        os.remove(os.path.join(d, "k_test_50.pt.meta"))                              # a cache made before the rule: no sidecar
        cached("k", build, 40, split="test", root=d, sig="v1")                       # adopted as valid as of now, reused
        adopted = len(calls) == n0 and os.path.exists(os.path.join(d, "k_test_50.pt.meta"))
        cached("k", build, 40, split="test", root=d, sig="v2")                       # the builder changed: outdated, rebuilt
        check("cached: a cache without a sidecar is adopted; another builder signature deletes and rebuilds",
              adopted and len(calls) == n0 + 1 and sorted(f for f in os.listdir(d) if f.startswith("k_test")) == ["k_test_40.pt", "k_test_40.pt.meta"])
        check("source_sig: stable for one function, different for another", source_sig(build) == source_sig(build) != source_sig(selftest))

        arr, p, done = dump("selftest", "a", (64, 32), np.float16, root=d)
        arr[:] = torch.arange(64 * 32).view(64, 32).numpy().astype(np.float16)
        arr.flush()
        before = open(p, "rb").read()
        mark_done(p)
        del arr
        arr2, p2, done2 = dump("selftest", "a", (64, 32), np.float16, root=d)
        check("dump: a complete dump is reused, not written again", (not done) and done2 and open(p, "rb").read() == before and arr2[1, 1] == 33)
        try:
            dump("selftest", "a", (64, 33), np.float16, root=d)
            wrong = False
        except ScratchError:
            wrong = True
        check("dump: the same key with another shape raises", wrong)

    guard(0)
    try:
        guard(10 ** 18)
        big = False
    except ScratchError:
        big = True
    try:
        guard(1, budget_gb=0.0)
        budget = False
    except ScratchError:
        budget = True
    check("guard: passes for nothing, refuses 1 EB (drive) and a zero budget", big and budget)
    print("%d of %d checks passed" % (sum(ok), len(ok)))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        print(__doc__)
