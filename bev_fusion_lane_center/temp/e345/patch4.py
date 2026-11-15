R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/"
p = R + "jprune_probe.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old[:80]
    s = s.replace(old, new, count)


# --- F7: atomic writes (calib, sweep.json)
rep('''def save_calib(calib, path):
    torch.save(dict(mean={k: v.cpu() for k, v in calib.mean.items()}, s={k: v.cpu() for k, v in calib.s.items()}, head={k: v.cpu() for k, v in calib.head.items()}, frames=calib.frames), path)
''', '''def atomic_write(path, data):
    """write bytes through a temporary file and `os.replace`: a crash leaves the old file or none, never a truncated one"""
    with open(path + ".tmp", "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def save_calib(calib, path):
    torch.save(dict(mean={k: v.cpu() for k, v in calib.mean.items()}, s={k: v.cpu() for k, v in calib.s.items()}, head={k: v.cpu() for k, v in calib.head.items()}, frames=calib.frames), path + ".tmp")
    os.replace(path + ".tmp", path)
''')

# --- F8: identity message
rep('''assert d_id == 0.0, f"an identity hook must reproduce the null to 0: {res}"''', '''assert d_id == 0.0, f"an identity hook must reproduce the null to 0 (restart with --deterministic): {res}"''')

# --- Sweep init: deterministic + atomic
rep('''    def __init__(self, run, dev, out_dir, runs_dir=None, max_frames=0, calib_frames=0, workers=2, val_stride=1, log=print):
        self.run, self.dev, self.out, self.log = run, dev, out_dir, log
        wait_for_gpu("jprune_probe sweep", dev)''', '''    def __init__(self, run, dev, out_dir, runs_dir=None, max_frames=0, calib_frames=0, workers=2, val_stride=1, deterministic=False, log=print):
        self.run, self.dev, self.out, self.log, self.deterministic = run, dev, out_dir, log, deterministic
        wait_for_gpu("jprune_probe sweep", dev)''')
rep('''        self.rec = json.load(open(self.path)) if os.path.exists(self.path) else dict(run=run, attach=bool(self.args.get("attach")), rows=self.rows, tokens=[t for b in self.val for t in b["token"]], calib_frames=self.calib.frames, passes={}, points=[], controls={})
        assert self.rec["rows"] == self.rows, "the sweep.json on disk is of other frames"

    def save(self):
        json.dump(self.rec, open(self.path + ".tmp", "w"))
        os.replace(self.path + ".tmp", self.path)
''', '''        self.rec = json.load(open(self.path)) if os.path.exists(self.path) else dict(run=run, attach=bool(self.args.get("attach")), deterministic=deterministic, rows=self.rows, tokens=[t for b in self.val for t in b["token"]], calib_frames=self.calib.frames, passes={}, points=[], controls={})
        assert self.rec["rows"] == self.rows, "the sweep.json on disk is of other frames"
        assert bool(self.rec.get("deterministic", False)) == deterministic or not self.rec["passes"], f"sweep.json was made with deterministic={self.rec.get('deterministic', False)}, this job runs with {deterministic}: the passes must share one setting"
        self.rec["deterministic"] = deterministic

    def save(self):
        atomic_write(self.path, json.dumps(self.rec).encode())
''')

# --- do(): gpu memory
rep('''        m, w = ablate(self.model, edits, self.taps)
        t = time.time()
        r, _ = run_pass(m, self.val, self.dev, self.half)
        w.check(r["forwards"], self.calib)
        sv = self.cost.saved(edits)
        r.update(spec=spec, how=how, saved_macs=sv["macs"], saved_params=sv["params"], saved_macs_pyr=sv["macs_pyr"], seconds=time.time() - t)''', '''        m, w = ablate(self.model, edits, self.taps)
        reset_gpu_peak(self.dev)
        t = time.time()
        r, _ = run_pass(m, self.val, self.dev, self.half)
        w.check(r["forwards"], self.calib)
        sv = self.cost.saved(edits)
        r.update(spec=spec, how=how, saved_macs=sv["macs"], saved_params=sv["params"], saved_macs_pyr=sv["macs_pyr"], seconds=time.time() - t, gpu_mb=gpu_peak_mb(self.dev))''')

# --- nulls / run_all: deterministic from self
rep('''    def nulls(self, deterministic):
        """the null twice; above the cap the deterministic algorithms are set and both rerun, still above: stop (the probe is then not the stored pipeline)"""''', '''    def nulls(self):
        """the null twice; above the cap the deterministic algorithms are set (recorded in sweep.json) and both rerun, still above: stop (the probe is then not the stored pipeline)"""''')
rep('''            assert attempt == 0 and not deterministic, f"the two nulls differ by {abs(a - b):.4f} in ends_mean6 (cap {NULL_CAP}) with the deterministic setting: fix the probe"
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic = True
            for k in ("null", "null2"):
                del self.rec["passes"][k]''', '''            assert attempt == 0 and not self.deterministic, f"the two nulls differ by {abs(a - b):.4f} in ends_mean6 (cap {NULL_CAP}) with the deterministic setting: fix the probe"
            apply_deterministic()                                              # cuBLAS read its workspace setting at start: if this does not bring the spread down, restart with --deterministic
            self.deterministic = self.rec["deterministic"] = True
            for k in ("null", "null2"):
                del self.rec["passes"][k]''')
rep('''    def run_all(self, score_points=(), controls=True, deterministic=False):
        """the whole job: nulls, plumbing, the single points, the random / reverse draws; then the dumps of `score_points` [{id, set}]"""
        self.nulls(deterministic)''', '''    def run_all(self, score_points=(), controls=True):
        """the whole job: nulls, plumbing, the single points, the random / reverse draws; then the dumps of `score_points` [{id, set}]"""
        self.nulls()''')

# --- F7: dump_point
a = s.index("    def dump_point(self, pid, spec):")
b = s.index("def wait_for_gpu(who, dev):")
s = s[:a] + '''    def dump_point(self, pid, spec):
        """the point's `ends_val.pkl` + `point.json` in `runs/e345_p<NNN>_<model>/` with a symlink to the unmodified model.pt (the stored run's folder gets no file). Every file goes through a temporary file and `os.replace`, `point.json` (with the sha256 of the pickle) LAST:
        a folder without it is incomplete and redone; a complete one is never overwritten (the same point set again is a no-op, another set under that number or a pickle that does not match its point.json is refused)"""
        d = os.path.join(self.runs_dir, f"e345_p{pid:03d}_{self.run}")
        if os.path.exists(os.path.join(d, "point.json")):
            meta = json.load(open(os.path.join(d, "point.json")))
            assert meta["spec"] == spec, f"{d} holds another point set: a point folder is never overwritten"
            assert hashlib.sha256(open(os.path.join(d, "ends_val.pkl"), "rb").read()).hexdigest() == meta["sha256"], f"{d}/ends_val.pkl does not match its point.json (truncated or changed): move the folder away and rerun"
            self.log(f"point {pid} is dumped already")
            return
        edits = spec_edits(spec, self.units, self.calib)
        m, w = ablate(self.model, edits, self.taps)
        frames, n = [], 0
        for out, b in forward_batches(m, self.val, self.dev):
            frames += frames_of(out, b)
            n += 1
        w.check(n, self.calib)
        os.makedirs(d, exist_ok=True)
        if not os.path.lexists(os.path.join(d, "model.pt")):
            os.symlink(os.path.abspath(os.path.join(self.runs_dir, self.run, "model.pt")), os.path.join(d, "model.pt"))
        blob = pickle.dumps(frames)
        atomic_write(os.path.join(d, "ends_val.pkl"), blob)
        atomic_write(os.path.join(d, "point.json"), json.dumps(dict(point=pid, spec=spec, run=self.run, frames=len(frames), sha256=hashlib.sha256(blob).hexdigest(), edits=[dict(kind=e.kind, tap=e.tap or e.block, idx=list(e.idx), head=e.head) for e in edits]), indent=1).encode())
        self.log(f"dumped point {pid} ({spec_name(spec)}) -> {d}")


''' + s[b:]

# --- F6: gate + memory
a = s.index("def wait_for_gpu(who, dev):")
b = s.index("def time_pass(run, dev, n_batches=4):")
s = s[:a] + '''GATE_TIMEOUT_S = 12 * 3600.0                                                          # these jobs wait for the GPU much longer than the chains do (`E345_GATE_TIMEOUT_S` changes it)


def wait_for_gpu(who, dev, timeout_s=None):
    """D100: a GPU step does not start next to two trainings. `gpu_gate.wait_for_room` waits (this job: 12 h, `E345_GATE_TIMEOUT_S`; the shared `GPU_GATE_TIMEOUT_S` of the chains is ignored here); when the GPU is still full afterwards the job EXITS with code 75 and a message
    instead of going on into a full GPU (exit 75 = a retry is wanted, as everywhere in this repository). No effect on the CPU."""
    if not str(dev).startswith("cuda"):
        return
    import gpu_gate
    timeout_s = float(os.environ.get("E345_GATE_TIMEOUT_S", timeout_s or GATE_TIMEOUT_S))
    shared = os.environ.pop("GPU_GATE_TIMEOUT_S", None)
    try:
        gpu_gate.wait_for_room(who=who, timeout_s=timeout_s)
    finally:
        if shared is not None:
            os.environ["GPU_GATE_TIMEOUT_S"] = shared
    try:
        limit = float(os.environ.get("GPU_GATE_MB", gpu_gate.DEFAULT_MB))
    except ValueError:
        limit = gpu_gate.DEFAULT_MB
    used = gpu_gate.used_mb()
    if limit > 0 and used is not None and used >= limit:
        print(f"[jprune_probe] {who}: {used:.0f} MiB of the GPU are still in use after {timeout_s:.0f} s (limit {limit:.0f}): not starting next to the trainings; run it again when the GPU is free", flush=True)
        sys.exit(75)


def reset_gpu_peak(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb(dev):
    """-> dict(max_allocated, max_reserved) of this process in MiB since the last reset (None on the CPU)"""
    if not str(dev).startswith("cuda"):
        return None
    return dict(max_allocated=torch.cuda.max_memory_allocated() / 2 ** 20, max_reserved=torch.cuda.max_memory_reserved() / 2 ** 20)


def apply_deterministic():
    """cuBLAS reads its workspace setting when it starts: call this BEFORE anything creates the CUDA context"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_deterministic(out_dir, requested):
    """the deterministic setting of a job: the one recorded in `sweep.json` (with a loud line when the command line says otherwise), a refusal of `--deterministic` against passes that were made without it"""
    p = os.path.join(out_dir, "sweep.json")
    if not os.path.exists(p):
        return requested
    rec = json.load(open(p))
    was = bool(rec.get("deterministic", False))
    if was and not requested:
        print("[jprune_probe] sweep.json was made with --deterministic: this resumed job runs with it too", flush=True)
        return True
    if requested and not was and rec.get("passes"):
        sys.exit(f"[jprune_probe] {p} holds passes made WITHOUT --deterministic: the passes must share one setting (use another --out or drop the flag)")
    return requested


''' + s[b:]

# time_pass: memory + cuda test
rep('''    wait_for_gpu("jprune_probe time", dev)
    model, _ = JEV.load_model(run, dev)''', '''    wait_for_gpu("jprune_probe time", dev)
    reset_gpu_peak(dev)
    model, _ = JEV.load_model(run, dev)''')
rep('''        fn()                                                                                         # warm-up
        if dev != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        fn()
        if dev != "cpu":
            torch.cuda.synchronize()
        t[name] = (time.time() - t0) / len(cache)
        print(f"{name}: {t[name]:.2f} s per batch of 8 on {dev} (the machine as it is now) -> {t[name] * 110:.0f} s per VAL pass")
    return t''', '''        fn()                                                                                         # warm-up
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        fn()
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        t[name] = (time.time() - t0) / len(cache)
        print(f"{name}: {t[name]:.2f} s per batch of 8 on {dev} (the machine as it is now) -> {t[name] * 110:.0f} s per VAL pass")
    mem = gpu_peak_mb(dev)
    if mem:
        print(f"GPU memory of this process: peak allocated {mem['max_allocated']:.0f} MiB, peak reserved {mem['max_reserved']:.0f} MiB")
    t["gpu_mb"] = mem
    return t''')

# --- CLI
rep('''        if a.deterministic:                                                  # cuBLAS reads its workspace setting at start: set it before anything touches the GPU
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
        os.makedirs(a.out, exist_ok=True)
        sw = Sweep(a.run, a.device, a.out, a.runs_dir, a.max, a.calib_max, a.workers, a.val_stride)
        sw.run_all(json.load(open(a.score_points)) if a.score_points else (), not a.no_controls, a.deterministic)''', '''        os.makedirs(a.out, exist_ok=True)
        det = resolve_deterministic(a.out, a.deterministic)
        if det:
            apply_deterministic()                                            # before the GPU is touched: cuBLAS reads its workspace setting at start
        sw = Sweep(a.run, a.device, a.out, a.runs_dir, a.max, a.calib_max, a.workers, a.val_stride, det)
        sw.run_all(json.load(open(a.score_points)) if a.score_points else (), not a.no_controls)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
