
# ------------------------------------------------------------------------------------------------ G
print("G: wiring")
stu = dict(camera_student="stub", student_trainable=True)
check(not raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="all", **stu)))), "check_config: student_frames 'all' with a student is accepted")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="all"))), "student_frames", "without camera_student"), "check_config: student_frames 'all' without camera_student is refused")
check(raises(ValueError, lambda: check_config(ModelConfig(**dict(DUAL, student_frames="oldest", **stu))), "student_frames", "'all'"), "check_config: an unknown student_frames is refused and the kinds are named")
findings = lambda mc, **kw: T.student_findings(T.TrainConfig(name="x", device="cpu", feature_cache="c", lane_images="i", **kw), ModelConfig(**dict(DUAL, **mc)))  # noqa: E731
check(any("n_cam >= 2" in m for m in findings(dict(stu, student_frames="all"), n_cam=1)) and not any("n_cam" in m for m in findings(dict(stu, student_frames="all"), n_cam=3)),
      "student_findings: student_frames 'all' with n_cam 1 is reported, n_cam 3 is fine")
check(findings(stu, img_jitter=0.5, img_jitter_from=0.25) == [] and any("needs model.camera_student" in m for m in findings({}, img_jitter=0.5)) and findings({}, img_jitter=0.0) == [],
      "student_findings: image jitter with a student is fine, without one it is reported, off needs nothing")
check(all(any("img_jitter" in m for m in findings(stu, **kw)) for kw in (dict(img_jitter=-0.1), dict(img_jitter=2.5), dict(img_jitter_from=1.0), dict(img_jitter_from=-0.1))),
      "student_findings: img_jitter outside 0 .. 2 and img_jitter_from outside [0, 1) are reported")
check(raises(ValueError, lambda: T.build(T.TrainConfig(name="x", device="cpu", feature_cache="c1", feature_cache2="c2", lane_images="img", n_cam=1, model=dict(DUAL, student_frames="all", **stu))), "n_cam >= 2")
      and isinstance(T.build(T.TrainConfig(name="x", device="cpu", feature_cache="c1", feature_cache2="c2", lane_images="img", img_jitter=0.5, model=dict(DUAL, student_frames="all", **stu)))[0].student, Stub),
      "train.build: 'all' with n_cam 1 is refused, the complete config with image jitter builds")
d = T.TrainConfig()
check(d.img_jitter == 0.0 and d.img_jitter_from == 0.0 and "img_jitter" in T.RESUME_SWITCHES and "img_jitter" not in T.RESUME_FREE and "img_jitter_from" not in T.RESUME_FREE,
      "TrainConfig: both default off, img_jitter is a resume switch, neither may change on a resume")
seen_kw = []


class RecDS:
    def __init__(self, *a, **k):
        seen_kw.append(k.get("lane_images_older"))


fake_nusc = types.SimpleNamespace(scene=[], get=lambda *a: {"name": "x"})
real = (T.CachedFrontRadarDataset, T.FeatureStore, T.available_scenes)
prev_mods = {k: sys.modules.get(k) for k in ("nuscenes", "nuscenes.nuscenes")}
sys.modules["nuscenes"], sys.modules["nuscenes.nuscenes"] = types.ModuleType("nuscenes"), types.SimpleNamespace(NuScenes=lambda *a, **k: fake_nusc)
T.CachedFrontRadarDataset, T.FeatureStore, T.available_scenes = RecDS, lambda p: None, lambda n, s: []
try:
    for frames in ("all", "current"):
        T.make_datasets(T.TrainConfig(name="x", device="cpu", feature_cache="c", feature_cache2="c2", lane_images="img", model={**DUAL, "camera_student": "stub", "student_frames": frames}))
    T.make_datasets(T.TrainConfig(name="x", device="cpu", feature_cache="c", model=dict(BASE)))
finally:
    T.CachedFrontRadarDataset, T.FeatureStore, T.available_scenes = real
    for k, v in prev_mods.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
check(seen_kw == [True, True, False, False, False, False], f"train.make_datasets asks the datasets for the older images exactly with a student and student_frames 'all' (train, val: {seen_kw})")
zeros = torch.zeros
contract = {"K": zeros(2, 3, 3, 3), "T_cam_car": zeros(2, 3, 4, 4), "ida": None, "v": zeros(2), "omega": zeros(2), "steps": [[] for _ in range(3)], "tokens": ["a", "b"],
            "radar": dict({f: zeros(5) for f in IF.RADAR_FIELDS}, b=torch.tensor([0, 0, 1, 1, 1])), "prob": zeros(2, 3, 4, 2, 3), "ctx": zeros(2, 3, 80, 2, 3), "hw": (16, 24)}
check(IF.check_inputs(contract) == [] and IF.check_inputs(dict(contract, lane_img_old=zeros(2, 2, 3, 4, 4), img_jitter=AU.draw_jitter(2, 1.0, 0))) == []
      and any("lane_img_old" in e for e in IF.check_inputs(dict(contract, lane_img_old=zeros(2, 3, 3, 4, 4)))) and any("lane_img_old" in e for e in IF.check_inputs(dict(contract, lane_img_old=zeros(3, 2, 3, 4, 4)))),
      "interface: lane_img_old (B, F - 1, 3, H, W) and img_jitter are known inputs; a wrong number of older frames or of samples is named")
from evaluation import bev_adapter as A_  # noqa: E402
from evaluation import bev_modules as BM_  # noqa: E402
from evaluation import bev_probe as BP_  # noqa: E402
on_c = build(camera_student="stub", student_trainable=True, student_frames="all").eval()
bc = data(3)
with torch.no_grad():
    base_c, zero_c = on_c.camera_bevs(bc), on_c.camera_bevs(BM_.zero_cam(bc))
    zero_cache_only = on_c.camera_bevs({**bc, **{k: bc[k] * 0 for k in ("prob", "ctx", "prob2", "ctx2", "lane_img")}})
check(all(not torch.equal(zero_c[:, k], base_c[:, k]) for k in range(3)) and not torch.equal(zero_c[:, 1:], zero_cache_only[:, 1:]),
      "the evaluation's zero_cam changes the BEV of EVERY frame of an 'all' model (a zeroing that left lane_img_old in would not)")
pb = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in bc.items()}
BP_.perturb(pb, "zero_cam")
pb2 = {"tokens": ["a", "b"], "lane_img": bc["lane_img"].clone(), "lane_img_old": bc["lane_img_old"].clone()}
BP_.perturb(pb2, "shuffle_cam")
check(int(pb["lane_img_old"].sum()) == 0 and int(pb["lane_img"].sum()) == 0 and torch.equal(pb2["lane_img_old"], torch.roll(bc["lane_img_old"], 1, 0)), "bev_probe.perturb: zero_cam zeroes lane_img_old, shuffle_cam rolls it over the samples")
item_c = lambda v: {"frames": [{k: torch.full((2,), float(v)) for k in A_.CAMERA_FRAME_KEYS} | {"token": "t"} for _ in range(3)], "lane_img": torch.full((3, 2), v, dtype=torch.uint8),  # noqa: E731
                    "lane_img_old": torch.full((2, 3, 2), v, dtype=torch.uint8)}
it_, other_ = item_c(5), item_c(9)
A_.perturb_camera(it_, other_, "zero_cam")
it2 = item_c(5)
A_.perturb_camera(it2, other_, "shuffle_cam")
check(int(it_["lane_img"].sum()) == 0 and int(it_["lane_img_old"].sum()) == 0 and int(it2["lane_img_old"][0, 0, 0]) == 9, "bev_adapter.perturb_camera: zero_cam zeroes lane_img_old of a dataset item, shuffle_cam replaces it with the other item's")

# ------------------------------------------------------------------------------------------------ H
print("H: planted bugs")


def inst(name, fn):
    """a hook for section H: every model build() returns gets its method `name` replaced by fn(original bound method, self, *args)"""
    def hook(m):
        orig = getattr(m, name)
        setattr(m, name, types.MethodType(lambda self, *a, **k: fn(orig, self, *a, **k), m))
    return hook


def older_with_graph(self, img):
    B, K = img.shape[:2]
    out = self.student(img.flatten(0, 1))                                          # the bug: no torch.no_grad()
    return {k: v.reshape(B, K, *v.shape[1:]) for k, v in out.items() if k.startswith(("prob", "ctx"))}


def current_without_graph(self, img):
    with torch.no_grad():
        return self.student(img)


def jit_in_eval(orig, self, batch, Fn):
    was, self.training = self.training, True                                       # the bug: the jitter does not look at the mode
    try:
        return orig(batch, Fn)
    finally:
        self.training = was


def student_on_frame_one(orig, self, batch, Fn):
    cur, old = orig(batch, Fn)
    return (cur, old) if old is None else (old[:, 0], old)


real_jitter = AU.jitter_images


def jitter_per_frame(img, j, salt=0):                                              # the bug: another draw for every frame of a sample
    if img.dim() != 5:
        return real_jitter(img, j, salt)
    return torch.stack([real_jitter(img[:, k], dict(j, u=j["u"].roll(k + 1, 1)), salt) for k in range(img.shape[1])], 1)


def jitter_unseeded_noise(img, j, salt=0):                                         # the bug: the pixel noise from the global random state
    old, AU.JITTER_NOISE = AU.JITTER_NOISE, 0.0
    try:
        x = real_jitter(img, j, salt)
    finally:
        AU.JITTER_NOISE = old
    return (x + 255 * AU.JITTER_NOISE * j["strength"] * torch.randn_like(x)).clamp(0, 255)


real_clip, real_collate = T.run_clip, FC.collate_cached
real_records = LB.older_records


def collate_flipped(items):
    b = real_collate(items)
    return dict(b, lane_img_old=b["lane_img_old"].flip(1)) if "lane_img_old" in b else b


def always_from_start(m_, l_, c_, f_, d_, a_, cfg_, **k):                          # the bug: img_jitter_from ignored
    return real_clip(m_, l_, c_, f_, d_, a_, dataclasses.replace(cfg_, img_jitter_from=0.0), **k)


# name: (patches [(object, attribute, new)], hook for every model, groups to try in this order)
MODEL_G, JIT_G = ("model", "stream"), ("jfn", "jmodel", "clip")
MUTANTS = {
    "the older frames' student forward keeps its graph (no torch.no_grad())": ([], inst("_run_student_older", lambda orig, self, img: older_with_graph(self, img)), MODEL_G),
    "the newest frame's student forward has no graph": ([], inst("_run_student", lambda orig, self, img: current_without_graph(self, img)), MODEL_G),
    "the older frames in the wrong order": ([], inst("_run_student_older", lambda orig, self, img: orig(img.flip(1))), MODEL_G),
    "the older images of the wrong sample": ([], inst("_run_student_older", lambda orig, self, img: orig(img.roll(1, 0))), MODEL_G),
    "the student reads frame 1's image for frame 0 (and so does the anchor)": ([], inst("_student_images", student_on_frame_one), MODEL_G),
    "the older images are ignored (frames >= 1 stay the cached teachers')": ([], inst("_student_images", lambda orig, self, batch, Fn: (orig(batch, Fn)[0], None)), MODEL_G),
    "the jitter does not look at the training mode": ([], inst("_student_images", jit_in_eval), JIT_G),
    "the jitter is drawn per frame instead of per sample": ([(AU, "jitter_images", jitter_per_frame), (FC, "jitter_images", jitter_per_frame)], None, JIT_G),
    "the pixel noise comes from the global random state": ([(AU, "jitter_images", jitter_unseeded_noise), (FC, "jitter_images", jitter_unseeded_noise)], None, JIT_G),
    "the draw comes from the global random state, not from (seed, step, micro)": ([(T, "draw_jitter", lambda B, s, seed, key=0: {"u": torch.rand(4, B) * 2 - 1, "strength": float(s), "noise_seed": int(seed) * 64 + key})], None, JIT_G),
    "the draw ignores the step": ([(T, "jitter_seed", lambda seed, step, micro: int(seed) * 1009 + int(micro))], None, JIT_G),
    "img_jitter_from is ignored (the jitter is on from step 0)": ([(T, "run_clip", always_from_start)], None, JIT_G),
    "the brightness / contrast / saturation range is 0.2, not 0.4": ([(AU, "JITTER_RANGE", 0.2)], None, JIT_G),
    "the gamma range is 1.5, not 1.25": ([(AU, "JITTER_GAMMA", 1.5)], None, JIT_G),
}
if HAVE_MINI:
    MUTANTS.update({
        "collate_cached drops lane_img_old": ([(FC, "collate_cached", lambda items: {k: v for k, v in real_collate(items).items() if k != "lane_img_old"})], None, ("data",)),
        "collate_cached stacks the older images in the wrong order": ([(FC, "collate_cached", collate_flipped)], None, ("data",)),
        "the builder caches the key frame again instead of the older frames": ([(LB, "older_records", lambda ds, t: [ds.gt.cam(t)] * (ds.n_cam - 1))], None, ("data",)),
        "the older frames go into the key-frame folder (no sweeps/)": ([(LB, "older_root", lambda root: root)], None, ("data",)),
        "the marker covers every scene": ([(LB, "older_missing_scenes", lambda *a: []), (ST, "older_missing_scenes", lambda *a: [])], None, ("marker",)),
    })
GROUPS = {"model": model_checks, "stream": stream_batch_checks, "jfn": jitter_fn_checks, "jmodel": jitter_model_checks, "clip": clip_checks}
if HAVE_MINI:
    GROUPS.update({"data": data_checks, "marker": marker_checks})


@contextlib.contextmanager
def planted(patches, hook):
    saved = [(o, a, getattr(o, a)) for o, a, _ in patches]
    for o, a, new in patches:
        setattr(o, a, new)
    MUTATE[0] = hook
    try:
        yield
    finally:
        MUTATE[0] = None
        for o, a, old in saved:
            setattr(o, a, old)


survived, killed = [], 0
for name, (patches, hook, groups) in MUTANTS.items():
    why = None
    with planted(patches, hook):
        for g in groups:
            try:
                bad = [k for k, v in GROUPS[g]().items() if not v]
            except Exception as e:                                                  # noqa: BLE001 - a planted bug that crashes a check is caught as well
                bad = [f"raised {type(e).__name__}: {e}"]
            if bad:
                why = f"{g}: {bad[0][:110]}" + (f" (+{len(bad) - 1})" if len(bad) > 1 else "")
                break
    check(why is not None, f"mutant killed: {name} -> {why or 'SURVIVED'}")
    killed += why is not None
    if why is None:
        survived.append(name)
print(f"  planted bugs killed: {killed} of {len(MUTANTS)}" + (f"; survived: {survived}" if survived else ""))
check(all(v for g in ("model", "jfn") for v in GROUPS[g]().values()), "after the planted bugs the unmutated code passes again")
