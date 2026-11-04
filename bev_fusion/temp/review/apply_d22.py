"""D22 (a): the six fixes of the research tools, with their selftest cases (spec: autoresearch/history/d22_tool_fixes.md)."""
import os
import re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
T = f"{ROOT}/autoresearch/tools"


def edit(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == count, (path, old[:70], s.count(old))
    open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    print("edited", os.path.relpath(path, ROOT))


# ---- 1 clean_temp.py: --delete only when the scratch really is kept in git
edit(f"{T}/clean_temp.py",
     '''def modules_in(folders):''',
     '''def kept_by_snapshot(cfg):
    """is the scratch folder inside one of the code_dirs that git_snapshot commits? (otherwise the tag does not hold the scripts)"""
    s = cfg["scratch"].strip("/").replace("\\\\", "/")
    return any(s == d.strip("/") or s.startswith(d.strip("/") + "/") for d in cfg["code_dirs"])


def modules_in(folders):''')
edit(f"{T}/clean_temp.py",
     '''        commit, tag, note = git_snapshot(root, cfg, cfg["tag_prefix"], f"{name}-scratch", f"scratch of {name} before clean-up")
        print(f"  code kept in git: {commit} {tag} {note}")
''',
     '''        commit, tag, note = git_snapshot(root, cfg, cfg["tag_prefix"], f"{name}-scratch", f"scratch of {name} before clean-up")
        print(f"  code kept in git: {commit} {tag} {note}")
        if delete and not tag:                     # no repository, or a git step failed: nothing holds the scripts
            print(f"  REFUSED: --delete needs the scratch in git, but the snapshot gave no tag ({commit} {note}); use --move or fix git")
            return 1
        if delete and not kept_by_snapshot(cfg):   # the snapshot commits code_dirs only
            print(f"  REFUSED: --delete needs '{cfg['scratch']}' in project.json code_dirs (the tag would not hold the scripts); use --move")
            return 1
''')
edit(f"{T}/clean_temp.py",
     '''    shutil.rmtree(root)
    print("clean_temp selftest passed")''',
     '''    # --delete refuses when nothing keeps the scripts: no repository here (git_snapshot gives no tag) ...
    os.makedirs(os.path.join(root, "temp/loop98"))
    open(os.path.join(root, "temp/loop98/a.py"), "w").write("x = 2\\n")
    os.utime(os.path.join(root, "temp/loop98/a.py"), (old, old))
    cfg98 = dict(cfg, code_dirs=["temp"])
    assert run("loop98", delete=True, root=root, cfg=cfg98, git=True) == 1 and os.path.exists(os.path.join(root, "temp/loop98/a.py")), \\
        "--delete without a tag must refuse and keep the scratch"
    # ... or when the scratch folder is not in code_dirs although a tag exists (stand-in for git_snapshot)
    import run as RUNMOD
    real = RUNMOD.git_snapshot
    RUNMOD.git_snapshot = lambda *a, **k: ("abc123", "t/loop98-scratch", "")
    try:
        assert run("loop98", delete=True, root=root, cfg=dict(cfg, code_dirs=["training"]), git=True) == 1 and os.path.exists(os.path.join(root, "temp/loop98/a.py")), \\
            "--delete with the scratch outside code_dirs must refuse"
        assert run("loop98", delete=True, root=root, cfg=cfg98, git=True) == 0 and not os.path.exists(os.path.join(root, "temp/loop98")), \\
            "--delete with a tag and the scratch in code_dirs goes through"
    finally:
        RUNMOD.git_snapshot = real
    shutil.rmtree(root)
    print("clean_temp selftest passed")''')

# ---- 2 project.py: a folder without project.json is not a problem
edit(f"{T}/project.py", "import os\nimport time\n", "import os\nimport sys\nimport time\n")
edit(f"{T}/project.py",
     '''    return os.path.normpath(p if os.path.isabs(p) else os.path.join(AR, p))
''',
     '''    r = os.path.normpath(p if os.path.isabs(p) else os.path.join(AR, p))
    if not os.path.isfile(os.path.join(r, "project.json")) and not os.path.isabs(p):
        alt = os.path.normpath(os.path.abspath(p))              # a relative AR_PROJECT may also mean: relative to the current folder
        if os.path.isfile(os.path.join(alt, "project.json")):
            r = alt
    if not os.path.isfile(os.path.join(r, "project.json")):    # defaults + a ghost ledger folder would follow otherwise
        sys.exit(f"no project.json in {r} (AR_PROJECT={os.environ.get('AR_PROJECT')!r}, else default_project of autoresearch/config.json): "
                 "that folder is not a problem folder")
    return r
''')

# ---- 3 run.py: run records and slot owner files are written atomically
edit(f"{T}/run.py",
     '''def record(d):
    try:
        return json.load(open(os.path.join(d, "run.json")))
    except Exception:
        return {}
''',
     '''def record(d):
    try:
        return json.load(open(os.path.join(d, "run.json")))
    except Exception:
        return {}


def save_rec(path, rec, indent=1):
    """write a run record / slot owner file through a temporary file and an atomic replace: other tools poll these files (state_of, schedule,
    tick, GpuSlot.stale); a torn read gave record() == {}, and a live slot with an unreadable owner file looked stale and was taken twice"""
    P.atomic_write(path, json.dumps(rec, indent=indent))
''')
s = open(f"{T}/run.py", encoding="utf-8").read()
old = 'json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1)'
assert s.count(old) == 5, s.count(old)
s = s.replace(old, 'save_rec(os.path.join(d, "run.json"), rec)')
old2 = 'json.dump(dict(id=self.id, pid=os.getpid(), run_dir=self.run_dir, taken=now()), open(os.path.join(s, "run.json"), "w"))'
assert s.count(old2) == 1
s = s.replace(old2, 'save_rec(os.path.join(s, "run.json"), dict(id=self.id, pid=os.getpid(), run_dir=self.run_dir, taken=now()), indent=None)')
open(f"{T}/run.py", "w", encoding="utf-8", newline="").write(s)
print("edited autoresearch/tools/run.py (6 writes)")

# ---- 4 run.py: an unreadable GPU is not free
edit(f"{T}/run.py",
     '''def gpu_info():''',
     '''def mem_ok(used_mb, max_mb):
    """the memory gate of --gpu-max-mb: no gate when max_mb is 0; a GPU that cannot be read (nvidia-smi failed: used_mb None) is NOT free"""
    return not max_mb or (used_mb is not None and used_mb < max_mb)


def gpu_info():''')
edit(f"{T}/run.py", "if (u is None or u < a.gpu_max_mb) and slot.try_take():", "if mem_ok(u, a.gpu_max_mb) and slot.try_take():")

# ---- 5 schedule.py: add refuses an existing item
edit(f"{T}/schedule.py",
     '''    else:
        res = a.resource or {"train": "gpu", "evaluate": "cpu"}.get(a.kind, "agent")''',
     '''    else:
        if any(r["item"] == a.item for r in rows):
            sys.exit(f"item {a.item} exists already in the schedule (change its state with `set`)")
        res = a.resource or {"train": "gpu", "evaluate": "cpu"}.get(a.kind, "agent")''')

# ---- 6 ledger.py: note / add need an id
edit(f"{T}/ledger.py",
     '''    a = ap.parse_args()
    if a.cmd == "render":''',
     '''    a = ap.parse_args()
    if a.cmd in ("note", "add") and not a.id:
        ap.error(f"{a.cmd} needs a run id (a row without an id would be written)")
    if a.cmd == "render":''')

# ---- selftest cases (before the clean_temp / guard_hook / metrics part)
edit(f"{T}/selftest.py",
     '''    import clean_temp
    clean_temp.selftest()''',
     '''    # D22: run records are written atomically (a reader never sees a torn file), an unreadable GPU is not free, a folder without
    # project.json is refused, schedule add refuses an existing item, ledger note needs an id
    import threading
    rd = f"{T}/runs/S.atomic"
    os.makedirs(rd, exist_ok=True)
    RUN.save_rec(os.path.join(rd, "run.json"), dict(id="S.atomic", pid=1))
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            RUN.save_rec(os.path.join(rd, "run.json"), dict(id="S.atomic", pid=1, notes="x" * 20000, i=i))
            i += 1
    th = threading.Thread(target=writer)
    th.start()
    try:
        torn = sum(1 for _ in range(3000) if not RUN.record(rd))
    finally:
        stop.set()
        th.join()
    assert torn == 0, f"{torn} torn reads of a run record"
    assert RUN.mem_ok(None, 9000) is False and RUN.mem_ok(None, 0) is True and RUN.mem_ok(8000, 9000) is True and RUN.mem_ok(9500, 9000) is False
    empty = tempfile.mkdtemp(prefix="ar_noproj_")
    r2 = subprocess.run([sys.executable, f"{HERE}/ledger.py", "render"], capture_output=True, text=True, env=dict(env, AR_PROJECT=empty))
    assert r2.returncode != 0 and "no project.json" in r2.stdout + r2.stderr and os.listdir(empty) == [], (r2.stdout, r2.stderr, os.listdir(empty))
    shutil.rmtree(empty)
    sch_before = open(f"{T}/schedule.csv", "rb").read()
    r5 = subprocess.run([sys.executable, f"{HERE}/schedule.py", "add", "S.race", "--kind", "train"], capture_output=True, text=True, env=env)
    assert r5.returncode != 0 and "exists already" in r5.stdout + r5.stderr and open(f"{T}/schedule.csv", "rb").read() == sch_before \\
        and not os.path.exists(f"{T}/schedule.csv.lock"), (r5.stdout, r5.stderr)
    led_before = open(f"{T}/LEDGER.csv", "rb").read()
    r6 = subprocess.run([sys.executable, f"{HERE}/ledger.py", "note", "--result", "x"], capture_output=True, text=True, env=env)
    assert r6.returncode != 0 and open(f"{T}/LEDGER.csv", "rb").read() == led_before, (r6.stdout, r6.stderr)
    import clean_temp
    clean_temp.selftest()''')

# ---- data: the planned duplicates of two running items
sp = f"{ROOT}/bev_fusion_lane_center/experiment_execution/schedule.csv"
lines = open(sp, encoding="utf-8", newline="").read().split("\n")
drop = [i for i, l in enumerate(lines) if l.startswith(("E240.a3,I2,train,planned,", "E240.c3,I2,train,planned,"))]
assert len(drop) == 2, drop
open(sp, "w", encoding="utf-8", newline="").write("\n".join(l for i, l in enumerate(lines) if i not in drop))
print("schedule.csv: removed the planned duplicates of E240.a3 / E240.c3 (lines", [d + 1 for d in drop], ")")
