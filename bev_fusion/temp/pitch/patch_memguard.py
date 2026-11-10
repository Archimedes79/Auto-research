def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_eval.py", [
    ('''    src._bev_subject = None
    _SOURCES.pop(getattr(src, "_bev_key", None), None)''', '''    src._bev_subject = None
    _SOURCES.pop(getattr(src, "_bev_key", None), None)
    import gc
    gc.collect()                                                        # a built network and its nuScenes tables are several GB: give them back before the next build'''),
])
patch("tests/test_evaluation_api.py", [
    ('''def real(run, scene, index):
    if not glob.glob(os.path.join(ROOT, run, "ckpt_*.pt")) or not os.path.isfile(os.path.join(ROOT, run, "config.json")):
        print(f"SKIP real check: {run} has no checkpoint here")
        return False''', '''def available_gb():
    try:
        with open("/proc/meminfo") as f:
            return int(next(ln for ln in f if ln.startswith("MemAvailable")).split()[1]) / 1e6
    except (OSError, StopIteration, ValueError):
        return None


def real(run, scene, index):
    if not glob.glob(os.path.join(ROOT, run, "ckpt_*.pt")) or not os.path.isfile(os.path.join(ROOT, run, "config.json")):
        print(f"SKIP real check: {run} has no checkpoint here")
        return False
    free = available_gb()
    if free is not None and free < NEED_GB:                                          # the box is shared: the real check holds two built networks and their tables (about 15 GB at its peak)
        print(f"SKIP real check: {free:.1f} GB available, it needs {NEED_GB} GB (wait until the other jobs are done)")
        return False'''),
    ('''ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))''', '''ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NEED_GB = 18'''),
])
print("ok")
