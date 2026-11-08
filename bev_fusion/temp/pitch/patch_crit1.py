p = "evaluation/bev_modules.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


# --- module list from the model (lift2, lane, cam_adapter ... are children too)
rep('''MODULES = ("lift", "cam_reduce", "cam_reduce2", "cam_merge", "cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "head_reduce",
           "motion_block", "sem_head", "obj_head", "aux_cam", "aux_rad")
''', '''MODULES = ("lift", "cam_reduce", "cam_reduce2", "cam_merge", "cam_temporal", "radar", "rad_temporal", "fusion", "trunk", "head_reduce",
           "motion_block", "sem_head", "obj_head", "aux_cam", "aux_rad")     # the chain of the t16 configuration, for reading; the probe takes the model's own children (chain_modules)
''')
rep('''def stat(t):''', '''def chain_modules(model):
    """the names of the model's direct child modules in definition order: everything the chain is made of (lift AND lift2 of a dual camera, lane, cam_adapter, ...),
    not a fixed list. aux_cam / aux_rad are children but only run while training (model.py), so they have no output in an evaluation-mode probe."""
    return [n for n, _ in model.named_children()]


def wrap_method(obj, name, wrapper):
    """replace obj.<name> by wrapper(original) on the instance; -> restore() that puts back exactly what was there (the class method, or the instance attribute)"""
    had = name in obj.__dict__
    original = getattr(obj, name)
    setattr(obj, name, wrapper(original))

    def restore():
        if had:
            setattr(obj, name, original)
        else:
            obj.__dict__.pop(name, None)
    return restore


def stat(t):''')
rep('''            out[prefix or "out"] = o.detach().float().cpu()''', '''            out[prefix or "out"] = o.detach().float().cpu().clone()               # a copy: float32 on the CPU is no copy, a later in-place op would change the record''')
rep('''    return {"shape": list(t.shape), "finite": bool(torch.isfinite(t).all()), "mean": float(t.mean()), "std": float(t.std()),''',
    '''    return {"shape": list(t.shape), "finite": bool(torch.isfinite(t).all()), "mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1 else 0.0,''')

# --- perturbations
rep('''def zero_cam(batch):
    """camera probability and context tensors times 0 (the camera features are cached: these are what the lift reads)"""
    return {**batch, **{k: batch[k] * 0 for k in ("prob", "ctx", "prob2", "ctx2") if k in batch and torch.is_tensor(batch[k])}}''',
    '''def zero_cam(batch):
    """every camera input times 0: the cached probability / context tensors the lift reads, and the images the lane branch reads (`lane_img`, `img`) - a camera
    path that is left in the batch would make the test say 'the camera does not matter' (the t3 cam_off bug)"""
    return {**batch, **{k: batch[k] * 0 for k in ("prob", "ctx", "prob2", "ctx2", "lane_img", "img") if k in batch and torch.is_tensor(batch[k])}}''')
rep('''def zero_doppler(batch):
    """the radial velocity of every detection 0: positions stay"""''', '''def zero_doppler(batch):
    """the radial velocity of every detection 0 (in the ego-compensated frame that reads 'every detection stands still'): positions, ranges and rcs stay"""''')

# --- response: warn when a perturbation changed nothing
rep('''def response_row(base_stages, base_out, new_stages, new_out):''', '''def changed_nothing(row):
    """True when no number of a response row moved: the perturbation did not reach the network (a key it did not remove, a flag that changed nothing)"""
    return all((v or 0.0) <= 1e-9 for d in row.values() for v in d.values())


def response_row(base_stages, base_out, new_stages, new_out):''')

# --- recorder: children, wrap_method
rep('''    def __init__(self, model, modules=MODULES):
        self.model, self.store, self._hooks, self._trunk = model, {}, [], None
        for name in modules:
            m = getattr(model, name, None)
            if m is not None and name != "trunk":
                self._hooks.append(m.register_forward_hook(self._hook(name)))
        trunk = getattr(model, "trunk", None)
        if trunk is not None and hasattr(trunk, "features"):
            self._trunk = (trunk, trunk.features)

            def features(*args, **kw):
                out = self._trunk[1](*args, **kw)
                self.store.setdefault("trunk", tensors_of(out))
                return out
            trunk.features = features
''', '''    def __init__(self, model, modules=None):
        self.model, self.store, self._hooks, self._restore = model, {}, [], None
        for name in (chain_modules(model) if modules is None else modules):
            m = getattr(model, name, None)
            if m is not None and name != "trunk":
                self._hooks.append(m.register_forward_hook(self._hook(name)))
        trunk = getattr(model, "trunk", None)
        if trunk is not None and hasattr(trunk, "features"):
            def wrapper(original):
                def features(*args, **kw):
                    out = original(*args, **kw)
                    self.store.setdefault("trunk", tensors_of(out))
                    return out
                return features
            self._restore = wrap_method(trunk, "features", wrapper)
''')
rep('''        if self._trunk is not None:
            self._trunk[0].features = self._trunk[1]
            self._trunk = None''', '''        if self._restore is not None:
            self._restore()
            self._restore = None''')

# --- probe: ckpt_dir, try/finally, warnings
rep('''def probe(run, frames=3, device="cpu", split="val", out_dir=None, threads=2):''', '''def probe(run, frames=3, device="cpu", split="val", out_dir=None, threads=2, ckpt_dir=None):''')
rep('''    src, subj, _ = A.build(run, device=device, split=split, workers=0)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg''', '''    src, subj, _ = A.build(run, device=device, split=split, workers=0, ckpt_dir=ckpt_dir)
    ns, model, loss, cfg = subj.ns, subj.model, subj.loss, subj.cfg''')
rep('''    rec = Recorder(model)
    refs = []
    for s in src.scenes():
        fr = src.frames(s)
        refs += [(s, len(fr) // 2), (s, len(fr) // 3)]
''', '''    rec = Recorder(model)
    refs = []
    for s in src.scenes():
        fr = src.frames(s)
        refs += [(s, len(fr) // 2), (s, len(fr) // 3)]
    if len(refs) < frames:
        print(f"WARNING: only {len(refs)} frames available (2 per scene), not {frames}", flush=True)
''')
rep('''    modules, response, rows = {}, {}, []
    for k, (scene, idx) in enumerate(refs[:frames]):''', '''    modules, response, rows = {}, {}, []
    try:
        _probe_frames(refs[:frames], src, subj, rec, forward, modules, response, rows, out_dir, run)
    finally:
        rec.close()
    return _write(modules, response, rows, out_dir)


def _probe_frames(refs, src, subj, rec, forward, modules, response, rows, out_dir, run):
    ns, model, cfg = subj.ns, subj.model, subj.cfg
    T, dev = ns.T, subj.device
    for k, (scene, idx) in enumerate(refs):''')
rep('''        for pname, perturb in PERTURBATIONS.items():
            r2 = forward(perturb(batch))
            response.setdefault(pname, []).append(response_row(base, out0, rec.copy(), r2["out"][0.0]))''', '''        for pname, perturb in PERTURBATIONS.items():
            r2 = forward(perturb(batch))
            row = response_row(base, out0, rec.copy(), r2["out"][0.0])
            if changed_nothing(row):
                print(f"WARNING: {pname} changed nothing in frame {ref.key}: the perturbation does not reach the network (a key it does not remove?)", flush=True)
            response.setdefault(pname, []).append(row)''')
rep('''        print("frame", k, ref.key, "done", flush=True)
    rec.close()
    summary = {k: mean_over_frames(v) for k, v in modules.items()}''', '''        print("frame", k, ref.key, "done", flush=True)


def _write(modules, response, rows, out_dir):
    summary = {k: mean_over_frames(v) for k, v in modules.items()}''')
rep('''    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("RUN is required (or --selftest)")
    print_report(probe(a.run, a.frames, a.device, a.split, a.out, a.threads))''', '''    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="the folder of a moved checkpoint")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not a.run:
        ap.error("RUN is required (or --selftest)")
    print_report(probe(a.run, a.frames, a.device, a.split, a.out, a.threads, a.ckpt_dir))''')

# --- docs
rep('''The probe module list (MODULES) is also the one `bev_timing.py` times. A module the network does not have (another configuration, an older snapshot) is
left out. Every perturbation of the response test is itself tested (`--selftest`): it changes exactly the keys it names and nothing else."""''',
    '''The probed modules are the model's direct children (`chain_modules`: lift AND lift2 of a dual camera, lane, ...), the trunk through its `.features()`; `bev_timing.py`
times the same. What to know when reading the numbers:
  * a module called once per dt (the heads) is recorded at its FIRST call (dt -0.5 for `sem_head:` / `obj_head:`); `OUT dt0:` holds the dt-0 outputs
  * `aux_cam` / `aux_rad` only run while training: no output here (the `radar_aux` tap of `qual debug` runs the head on the radar branch for a look)
  * the response row of a stage uses its first two tensors (motion_block: the two coarsest levels; `OUT:v_hat` is the finest)
  * `zero_doppler` sets the radial velocity of every detection to 0, which in the ego-compensated frame reads "every detection stands still", not "no Doppler";
    the camera perturbation also zeroes `lane_img` / `img`; a perturbation that changes nothing at all prints a WARNING (the t3 cam_off class of bug)
  * "static" cells of the velocity check are all defined cells with GT speed < 0.3 m/s, empty ground included, not only static objects
  * the velocity-field and response numbers of a few frames are a probe, not a measurement: no confidence interval
Every perturbation is itself tested (`--selftest`): it changes exactly the keys it names and nothing else."""''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
