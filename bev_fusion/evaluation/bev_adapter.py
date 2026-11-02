"""The BEV adapter of the evaluation suite (autoresearch/evalsuite): Source + Subject + Task of a training run `runs/<name>`.

    build(run="runs/t1_r4_soa", device="cuda", split="val", workers=2, ckpt_dir=None, control="", prior=0)  ->  (Source, Subject, Task)

    BEVSource      the val (or train) keyframes of the run's config, grouped into scenes (videos) in time order; one frame =
                   the cached-feature batch of the training code (batch of 1) + the GT of -0.5 / 0 / +0.5 s; worker processes read ahead;
                   describe() carries `gt_signature` (the GT identity: the snapshot pins the code, not autogt, which is read live) and
                   `evaluator_hash` (the evaluator's source files at build: the store's identity rule refuses a resume with other evaluator code)
    BEVSubject     the run's last checkpoint under its EMA weights (the run's own code snapshot `runs/<name>/code` when it exists, so
                   a run is evaluated with the code it was trained with), the chain's state carried frame to frame along a scene,
                   forward hooks for the taps
    BEVPriorSubject  the strong simple baseline: the per-cell mean of the GT over n train keyframes as a constant prediction
    BEVTask        evaluation/bev_task.py (metrics, views, checks)
Pinned code: `setup_paths(run)` puts the snapshot first on sys.path BEFORE anything of the chain is imported; one run per process.
Controls (sanity: the model must lose what the control takes away): control = shuffle_radar | zero_radar | shuffle_cam | zero_cam.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
AUTORESEARCH = os.path.abspath(os.path.join(ROOT, "..", "autoresearch"))
CAMERA_FRAME_KEYS = ("prob", "ctx", "img", "prob2", "ctx2")     # the camera inputs of a frame dict (the camera controls replace all of them)


def perturb_camera(item, other, control):
    """in place, the camera sanity controls on one dataset item: EVERY camera input replaced by `other`'s (shuffle_cam) or zeroed
    (zero_cam): the frames' CAMERA_FRAME_KEYS (both feature caches of t10 "dual", the image) and the lane branch's key-frame image (t7).
    Before 2026-10-31 only prob / ctx / img were replaced, so t7 / t10 controls kept camera information. An unknown camera-like key
    is refused rather than left in."""
    for f, o in zip(item["frames"], other["frames"]):
        unknown = [k for k in f if k.startswith(("prob", "ctx", "img")) and k not in CAMERA_FRAME_KEYS]
        if unknown:
            raise ValueError(f"camera control: frame keys {unknown} are not handled (add them to CAMERA_FRAME_KEYS)")
        for k in CAMERA_FRAME_KEYS:
            if k in f:
                f[k] = o[k] if control == "shuffle_cam" else f[k] * 0
    if "lane_img" in item:
        item["lane_img"] = other["lane_img"] if control == "shuffle_cam" else item["lane_img"] * 0


def setup_paths(run: Optional[str] = None) -> str:
    """the run's code snapshot (runs/<name>/code) or training/ first on sys.path, then the bev_fusion folder and autoresearch;
    returns the code folder"""
    code = os.path.abspath(os.path.join(run, "code")) if run and os.path.isdir(os.path.join(run, "code")) else os.path.join(ROOT, "training")
    for p in (AUTORESEARCH, ROOT, code):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return code


def _chain():
    """the chain's code of the current snapshot (imported after setup_paths)"""
    import train as T
    from losses import NUSC_GROUPS, sample_targets
    from object_head import decode
    ns = SimpleNamespace(T=T, NUSC_GROUPS=NUSC_GROUPS, sample_targets=sample_targets, decode=decode,
                         clip_step=getattr(T, "clip_step", None), detach_tree=getattr(T, "detach_tree", None))
    try:
        from camera_encoder import depth_centres
        ns.depth_centres = lambda: depth_centres().numpy()
    except Exception:                                                         # noqa: BLE001
        ns.depth_centres = None
    return ns


# ------------------------------------------------------------------------------------------------ loading
def load_run(run: str, device: str, ckpt_dir: Optional[str] = None, ns=None):
    """(cfg, model, loss, step, checkpoint path): the last checkpoint under its EMA weights"""
    import torch
    ns = ns or _chain()
    T = ns.T
    with open(os.path.join(run, "config.json")) as f:
        cfg = T.TrainConfig(**json.load(f))
    model, loss, _ = T.build(cfg)
    model, loss = model.to(device), loss.to(device)
    if getattr(cfg, "train_only", ()) or getattr(cfg, "freeze", ()):     # train() applies the plan before it builds the optimiser: its state has a slot per TRAINABLE parameter
        import freeze as freeze_mod                                     # the run's own snapshot first on sys.path (setup_paths)
        freeze_mod.apply(model, cfg.train_only, cfg.freeze)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir or run, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir or run}")
    opt = torch.optim.AdamW(T.param_groups(model, loss, cfg.lr, cfg.weight_decay), lr=cfg.lr)      # as train(), to load
    ema = T.EMA(model, cfg.ema)
    step, _ = T.load_checkpoint(ckpts[-1], model, loss, opt, ema)
    ema.apply(model)                                                         # the EMA weights are evaluated
    model.eval()
    return cfg, model, loss, step, ckpts[-1]


def freeze_scenes(cfg) -> None:
    """Runs of 2026-10-26 (t1_*) have no scene lists and meant the 428 / 82 scenes of nuScenes parts 1-6; after parts 7-10 were added
    the same config means 700 / 150 scenes and most val frames would have no cached features. An empty list is replaced by the
    frozen one (data/scenes_parts1_6.json, data/README.md)."""
    path = os.path.join(ROOT, "data", "scenes_parts1_6.json")
    if os.path.exists(path) and not cfg.val_scenes and cfg.feature_cache:
        frozen = json.load(open(path))
        cfg.train_scenes, cfg.val_scenes = cfg.train_scenes or frozen["train"], frozen["val"]


def gt_version(sig) -> Optional[str]:
    """the GT version key (D29: v5, ...) of a GT identity '<res>_<ss>_<generator signature>'; None when it is not a released version"""
    try:
        from autogt.versions import version_of_key
        return version_of_key(sig) if sig else None
    except ImportError:
        return None


def gt_signature(ds) -> Optional[str]:
    """identity of the GT a dataset scores against: '<res>_<ss>_<generator signature>' (the suffix of its GT cache files, the same as
    FrontRadarDataset.gt_signature of the current code); None for a code snapshot whose generator has no signature"""
    try:
        if hasattr(ds, "gt_signature"):                                       # one formula: the dataset's own (it holds the grid extents of a non-default grid)
            return ds.gt_signature()
        return f"{ds.grid.res}_{ds.gt_ss}_{ds.gt.signature()}"                # a code snapshot from before the method existed
    except Exception:                                                         # noqa: BLE001
        return None


PROVENANCE_DIR = os.path.join(ROOT, "..", "autoresearch", "tools")
EVALUATOR_PATHS = (os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "..", "autoresearch", "evalsuite"))        # everything the evaluator is made of (the card's provenance block)
# the files that shape the NUMBERS (the resume guard of a store): the metrics, the adapter that predicts, the task and its optional blocks, the evalsuite's frame loop,
# counts, sampling and comparison core. Not the selftests, mutants, views, board, report or README: an edit there changes no number and must not orphan a
# store that took 30 minutes while other chats work in the same folders
NUMBER_PATHS = tuple(os.path.join(ROOT, "evaluation", f) for f in ("bev_adapter.py", "bev_task.py", "metrics.py", "bev_velocity.py", "bev_fusion_area.py", "bev_curves.py")) + \
    tuple(os.path.join(ROOT, "..", "autoresearch", "evalsuite", f) for f in ("comparison", "quantitative.py", "store.py", "runner.py", "contracts.py", "sampling.py"))


def _provenance_module():
    """autoresearch/tools/provenance.py loaded by path (tools/ is not a package); the folder it puts first on sys.path is taken away again, so the run's pinned
    code snapshot keeps its place"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("provenance", os.path.join(PROVENANCE_DIR, "provenance.py"))
    mod = importlib.util.module_from_spec(spec)
    keep = list(sys.path)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = keep
    return mod


def evaluator_tree_hash(mod=None, paths=None) -> str:
    """12 hex over the source files of the evaluator (default: all of evaluation/ and autoresearch/evalsuite/), provenance.tree_hash"""
    mod = mod or _provenance_module()
    return mod.tree_hash([os.path.abspath(p) for p in (EVALUATOR_PATHS if paths is None else paths)], mod.repo_root())


def evaluator_hash() -> Optional[str]:
    """the hash of the files that shape the numbers (NUMBER_PATHS) for the Source's description (None, with a warning, when the collector cannot be loaded: a
    store started then differs from every other, so a resume is refused rather than mixed)"""
    try:
        return evaluator_tree_hash(paths=NUMBER_PATHS)
    except Exception as ex:                                                   # noqa: BLE001
        print(f"WARNING: evaluator hash not available ({type(ex).__name__}: {ex}); the store is not protected against a resume with other evaluator code", flush=True)
        return None


def layer_index(spec):
    return {name: (i, kind, key) for i, (name, kind, key) in enumerate(spec)}


def predictions(p, b: int, spec):
    """one sample's dict of probability maps (areas, lines, objects) and distance fields from the model output p"""
    import torch
    sem = p["sem"][b].float()
    prob, dist = {}, {}
    for name, (i, kind, key) in layer_index(spec).items():
        if kind in ("area", "object", "line"):
            prob[name] = torch.sigmoid(sem[i]).cpu().numpy()
        elif kind == "dist":
            dist[name[:-len("_dist")]] = sem[i].clamp(min=0).cpu().numpy()
    return prob, dist


def _identity(x):
    return x


# ------------------------------------------------------------------------------------------------ Source
def _make_source_class():
    from evalsuite import FrameRef, Sample, Source

    class _BEVSource(Source):
        def __init__(self, cfg, ds, ns, workers: int = 2, control: str = "", split: str = "val", evaluator: Optional[str] = None):
            self.cfg, self.ds, self.ns, self.workers, self.control, self.split = cfg, ds, ns, workers, control, split
            self.evaluator = evaluator                                    # evaluator_hash(), taken once at build: a resumed store must come from the same evaluator code
            self.nusc = ds.nusc
            self.fn = ns.T.collate_cached if cfg.feature_cache else ns.T.collate
            by: Dict[str, list] = {}
            for i, tok in enumerate(ds.samples):
                s = self.nusc.get("sample", tok)
                by.setdefault(self.nusc.get("scene", s["scene_token"])["name"], []).append((s["timestamp"], i, tok))
            self._frames, self._idx = {}, {}
            for sc, lst in by.items():
                lst.sort()
                self._frames[sc] = [FrameRef(sc, k, tok) for k, (_, i, tok) in enumerate(lst)]
                for _, i, tok in lst:
                    self._idx[tok] = i
            self._stream = None                                           # (iterator, refs, position)

        def scenes(self) -> List[str]:
            return list(self._frames)

        def frames(self, scene: str):
            return self._frames[scene]

        def describe(self) -> Dict[str, Any]:
            sig = gt_signature(self.ds)
            return {"class": "BEVSource", "dataset": "nuScenes " + self.cfg.version, "split": self.split, "scenes": len(self._frames),
                    "frames": len(self.ds), "feature_cache": self.cfg.feature_cache, "control": self.control or None,
                    "gt_signature": sig, "gt_version": gt_version(sig), "evaluator_hash": self.evaluator}

        # -- read-ahead by worker processes (the Runner announces the frames it will load, in order)
        def prefetch(self, refs: Sequence[FrameRef]) -> None:
            import torch
            self._stream = None
            if self.workers <= 0 or self.control:
                return
            sub = torch.utils.data.Subset(self.ds, [self._idx[r.key] for r in refs])
            dl = torch.utils.data.DataLoader(sub, batch_size=None, shuffle=False, num_workers=self.workers, collate_fn=_identity,
                                             prefetch_factor=2)
            self._stream = [iter(dl), list(refs), 0]

        def _item(self, ref: FrameRef):
            st = self._stream
            if st is not None and st[2] < len(st[1]) and st[1][st[2]].key == ref.key:
                st[2] += 1
                return next(st[0])
            return self.ds[self._idx[ref.key]]

        def _perturb(self, item, ref):
            """sanity controls: the radar / camera features of another frame, or nothing. The other frame lies half the split away, i.e. in another
            scene (the samples are in scene order: 7 keyframes later was the same street 3.5 s on, which kept the static layers and made the control weaker than it claims;
            sanity numbers recorded before 2026-11-01 used the +7 frame)"""
            c = self.control
            other = self.ds[(self._idx[ref.key] + len(self.ds) // 2) % len(self.ds)]
            if c == "shuffle_radar":
                item["radar"] = other["radar"]
            elif c == "zero_radar":
                item["radar"] = {k: v[:0] for k, v in item["radar"].items()}
            elif c in ("shuffle_cam", "zero_cam"):
                perturb_camera(item, other, c)
            else:
                raise ValueError(f"unknown control {c}")

        def load(self, ref: FrameRef) -> Sample:
            item = self._item(ref)
            if self.control:
                self._perturb(item, ref)
            batch = self.fn([item])
            gts = self.ns.sample_targets(batch["gt"])
            nusc, tok = self.nusc, ref.key

            def camera():
                from PIL import Image
                sd = nusc.get("sample_data", nusc.get("sample", tok)["data"]["CAM_FRONT"])
                return np.asarray(Image.open(os.path.join(nusc.dataroot, sd["filename"])).convert("RGB"))

            def camera_model():
                from PIL import Image
                sd = nusc.get("sample_data", nusc.get("sample", tok)["data"]["CAM_FRONT"])
                im = Image.open(os.path.join(nusc.dataroot, sd["filename"])).convert("RGB")
                hw = batch.get("hw", (256, 704))
                H, W = int(hw[0]), int(hw[1])
                try:
                    inv = np.linalg.inv(np.asarray(batch["ida"], np.float64))
                    return np.asarray(im.transform((W, H), Image.AFFINE, data=tuple(inv[:2].flatten()), resample=Image.BILINEAR))
                except Exception:                                          # noqa: BLE001
                    return np.asarray(im.resize((W, H)))
            return Sample(ref=ref, inputs=batch, reference={dt: gts[dt][0] for dt in gts},
                          display={"camera": camera, "camera_model": camera_model},
                          meta={"timestamp": item.get("timestamp"), "token": tok})
    return _BEVSource


# ------------------------------------------------------------------------------------------------ Subject
def bev_taps(grid):
    from evalsuite import TapSpec
    ext = (grid.y_min, grid.y_max, grid.x_min, grid.x_max)
    kw = dict(extent=ext, flip_x=True, optional=True)         # which stages exist depends on the configuration / snapshot
    return [TapSpec("lift", "feature_map", "camera", "energy only inside the camera cone (a wedge), strongest in the visible near field",
                    "energy outside the wedge, empty near field", **kw),
            TapSpec("cam_reduce", "feature_map", "camera", "the same wedge after the channel reduction; GT boxes sit on energy blobs",
                    "energy that does not follow the objects", **kw),
            TapSpec("cam_temporal", "feature_map", "camera", "the cone filled further and smoother after the temporal fusion",
                    "the past erasing the present", **kw),
            TapSpec("temporal_agg", "feature_map", "camera", "what the moved past frames supply: small where nothing moved",
                    "everything changes, or nothing at all", **kw),
            TapSpec("memory_conf", "map", "camera", "world-memory confidence: 0 on the first frame of a scene, grows as it runs",
                    "never grows", **{**kw, "alive_check": False}),
            TapSpec("camera_visibility", "map", "camera", "1 = the camera sees the cell", "all ones or all zeros", **kw),
            TapSpec("radar_raster", "feature_map", "radar", "detections per cell (first channel) as dots", "raster shifted against the objects", **kw),
            TapSpec("radar_feat", "feature_map", "radar", "the raster smeared by a 3x3 context (delta_slots: the features of the slot encoder)", "features without raster dots", **kw),
            TapSpec("radar_state", "feature_map", "radar", "delta_slots only: the lossless slot raster (valid, dr, dpsi, Doppler code, rcs, age, offsets per slot) next to the features: "
                    "blobs where the detections are", "empty although there are detections, or blobs away from the objects", **kw),
            TapSpec("radar_memory", "feature_map", "radar", "the radar memory (rad_temporal, when on): older sweeps moved to the present", "never changes from frame to frame", **kw),
            TapSpec("fused", "feature_map", "fusion", "camera and radar features together", "dead channels", **kw),
            TapSpec("trunk", "feature_map", "trunk", "feature regions: road, off-road, objects", "noise or one single colour",
                    extent=ext, flip_x=True),
            TapSpec("motion_l0", "feature_map", "motion", "motion block, coarsest level: |v| smooth, bright on moving objects", "uniform, or a field along the ego motion", **kw),
            TapSpec("motion_l1", "feature_map", "motion", "motion block, second level (the finest when the block has two): the coarse field refined", "noise, or the coarse field copied", **kw),
            TapSpec("motion_l2", "feature_map", "motion", "motion block, FINEST level whatever the number of levels (= v_hat): moving objects bright, static cells ~ 0",
                    "bright on static cells (ego motion not removed)", **kw),
            TapSpec("hurdle_mov", "logits", "heads", "the hurdle's moving / static logits per class (max sigmoid shown): high on moving objects", "high everywhere, or nowhere",
                    **{**kw, "alive_check": False}),
            TapSpec("radar_aux", "logits", "heads", "the radar auxiliary head on the radar branch (evaluation-time view; trained only with aux_heads): radar-visible layers", "empty", **kw)]


def output_taps(model, out0, arrays, names, device):
    """the taps that are not a module's output, added to `arrays` in place: `hurdle_mov` (the object head's moving logits per class of dt 0) and `radar_aux` (the radar
    auxiliary head, which only runs while training, applied here to the radar branch's output: the radar memory's when the memory is on, the radar encoder's
    otherwise; with the memory on but its tap unreadable the tap is skipped, not fed with the wrong stage)"""
    import torch
    if "hurdle_mov" in names and "mov" in out0["obj"]:
        arrays["hurdle_mov"] = out0["obj"]["mov"][0].float().cpu().numpy()
    if "radar_aux" in names and getattr(model, "aux_rad", None) is not None:
        feat = arrays.get("radar_memory") if getattr(model, "rad_temporal", None) is not None else arrays.get("radar_feat")
        if feat is not None:
            arrays["radar_aux"] = model.aux_rad(torch.from_numpy(feat)[None].to(device))[0].float().cpu().numpy()


def _first_of_pair(o):
    """the radar encoder returns (features, lossless state) for radar_kind 'delta_slots' (with_state=True), the features alone otherwise"""
    return o[0] if isinstance(o, tuple) else o


def _state_of_pair(o):
    """the lossless raster of a delta_slots radar encoder; anything else has none (the recorder skips a tap that cannot be read)"""
    return o[1] if isinstance(o, tuple) and len(o) > 1 else None


def bev_hooks() -> dict:
    """the forward hooks of the chain's stages (name -> evalsuite.torch_taps.Hook); recorded only when armed"""
    from evalsuite.torch_taps import Hook
    return {"lift": Hook("lift", pick=lambda o: o[:1], select="first"),
            "cam_reduce": Hook("cam_reduce", pick=lambda o: o[:1], select="first"),
            "cam_temporal": Hook("cam_temporal", pick=lambda o: o[0]),
            "temporal_agg": Hook("cam_temporal", pick=lambda o: o[2]["agg"]),
            "memory_conf": Hook("cam_temporal", pick=lambda o: o[2]["mem_conf"]),
            "radar_raster": Hook("radar.net", when="in"),
            "radar_feat": Hook("radar", pick=_first_of_pair),
            "radar_state": Hook("radar", pick=_state_of_pair),
            "radar_memory": Hook("rad_temporal", pick=lambda o: o[0]),
            "fused": Hook("fusion"),
            "motion_l0": Hook("motion_block", pick=lambda o: o[0]),
            "motion_l1": Hook("motion_block", pick=lambda o: o[1]),
            "motion_l2": Hook("motion_block", pick=lambda o: o[-1])}                    # the finest level, whatever the number of levels


def _make_subject_classes():
    from evalsuite import Prediction, Subject
    from evalsuite.torch_taps import TapRecorder

    hooks = bev_hooks()

    class _BEVSubject(Subject):
        def __init__(self, run, device, ns, ckpt_dir=None):
            import torch
            self.run, self.ns = run, ns
            self.device = device if torch.cuda.is_available() else "cpu"
            self.cfg, self.model, self.loss, self.step, self.ckpt = load_run(run, self.device, ckpt_dir, ns)
            self.warmup = max(0, int(self.cfg.seq_len) - 1)
            self.rec = TapRecorder(self.model, hooks)
            self.state, self.prev = None, None
            self.refine = bool(getattr(self.cfg, "model", {}).get("refine")) and hasattr(self.model, "boxes")

        def reset(self):
            self.state, self.prev = None, None

        def taps(self):
            return bev_taps(self.model.grid)

        def describe(self):
            return {"class": "BEVSubject", "name": os.path.basename(os.path.normpath(self.run)), "run": self.run,
                    "checkpoint": os.path.basename(self.ckpt), "step": self.step, "weights": "EMA", "device": self.device,
                    "seq_len": self.cfg.seq_len, "refine": self.refine,
                    "code": os.path.relpath(self.ns.code, ROOT)}

        def ops(self, sample):
            """GMACs of the cached chain at batch 1 with dts -0.5 / 0 / +0.5 (the path `benchmark` times), counted by torch; the pre-trained
            camera heads are a fixed cost that is not in it"""
            from evalsuite.torch_taps import count_ops
            b1 = self.ns.T.to_device(sample.inputs, self.device)
            r = count_ops(lambda: self.model(b1, dts=(-0.5, 0.0, 0.5)))
            r["note"] += "; the cached chain at batch 1 with dts -0.5 / 0 / +0.5 (what benchmark() times), without the pre-trained camera heads"
            return r

        def predict(self, sample, taps=()):
            import torch
            ns, T = self.ns, self.ns.T
            names = set(taps)
            want = set(names)
            if "radar_aux" in names:                                       # the aux head reads the radar branch: its input is recorded too (not returned)
                want |= {"radar_feat", "radar_memory"}
            self.rec.arm([n for n in want if n in hooks])
            batch = sample.inputs
            kw = {}
            if self.state is not None and ns.clip_step is not None and self.prev is not None:
                kw = {"state": self.state, "step": ns.clip_step(self.prev, self.device)}
            with torch.no_grad():
                out, res, _ = T.forward_loss(self.model, self.loss, batch, 1.0, self.device, self.cfg.amp, self.cfg, train_terms=False, **kw)
                if ns.detach_tree is not None and "state" in res:
                    self.state, self.prev = ns.detach_tree(res["state"]), batch
                grid, spec = self.model.grid, self.model.spec
                refined = ({dt: self.model.boxes(res, dt, k=100, threshold=0.05, rescore=True) for dt in res["out"]} if self.refine else {})
                dts = {}
                for dt, p in res["out"].items():
                    prob, dist = predictions(p, 0, spec)
                    dts[dt] = {"prob": prob, "dist": dist,
                               "mass": p["mass"][0, 0].float().cpu().numpy() if p.get("mass") is not None else None,
                               "dets": ns.decode({k: v[:1].float().cpu() for k, v in p["obj"].items()}, grid, ns.NUSC_GROUPS, k=100, threshold=0.05)[0],
                               "detr": list(refined[dt][0]) if self.refine else None,
                               "vhat": p["v_hat"][0].float().cpu().numpy() if p.get("v_hat") is not None else None,
                               "objvel": p["obj"]["vel"][0].float().cpu().numpy() if "vel" in p["obj"] else None}     # the object head's raw velocity map (bev_velocity centres)
                arrays = self.rec.read(0) if names else {}
                if "memory_conf" in arrays:
                    arrays["memory_conf"] = arrays["memory_conf"][0]
                if "trunk" in names:
                    arrays["trunk"] = res["features"][0].float().cpu().numpy()
                output_taps(self.model, res["out"][0.0], arrays, names, self.device)
                if "camera_visibility" in names and hasattr(self.model, "camera_visibility"):
                    vis = self.model.camera_visibility(T.to_device(batch, self.device), res["features"])
                    arrays["camera_visibility"] = vis[0, 0].float().cpu().numpy()
            loss = {k: float(v) for k, v in out.items() if k == "total" or k.startswith("group.")}
            return Prediction(outputs={"dt": dts}, taps={k: v for k, v in arrays.items() if k in names}, info={"loss": loss})

        def benchmark(self, sample, n=30, warmup=5):
            """ms per frame of the CACHED chain at batch 1 (the pre-trained camera heads are a fixed cost, timed separately):
            dt 0 and (-0.5, 0, +0.5) [+ refined boxes]; only valid on an idle GPU (the load is recorded by the caller)"""
            import torch
            from evaluation.metrics import ms_per_frame
            T = self.ns.T
            b1 = T.to_device(sample.inputs, self.device)
            out = {}
            with torch.no_grad():
                out["bev_dt0_median"] = ms_per_frame(lambda: self.model(b1, dts=(0.0,)), n=n, warmup=warmup, batch1=False)["median"]
                m3 = ms_per_frame(lambda: self.model(b1, dts=(-0.5, 0.0, 0.5)), n=n, warmup=warmup, batch1=False)
                out["bev_3dt_median"] = m3["median"]
                if self.refine:
                    out["bev_dt0_refined_boxes_median"] = ms_per_frame(lambda: self.model.boxes(self.model(b1, dts=(0.0,)), 0.0), n=n,
                                                                     warmup=warmup, batch1=False)["median"]
                try:
                    from camera_encoder import FrontCameraEncoder
                    cam = FrontCameraEncoder().to(self.device).eval()
                    x, cv = torch.randn(1, 3, 256, 704, device=self.device), torch.zeros(1, 27, device=self.device)
                    out["camera_heads_median"] = ms_per_frame(lambda: cam.image_features(x, cv), n=min(n, 30), warmup=5, batch1=False)["median"]
                    del cam
                except Exception as e:                                       # noqa: BLE001
                    out["camera_heads_error"] = str(e)
            out["median_ms"], out["p90_ms"], out["mean_ms"], out["n"] = m3["median"], m3["p90"], m3["mean"], n
            return out

        def close(self):
            self.rec.close()

    class _PriorSubject(Subject):
        """the constant prediction: per-cell mean of the GT layers over n train keyframes (what the position in the grid alone gives)"""
        warmup = 0

        def __init__(self, maps, spec, groups):
            self.prob, self.dist = maps
            self.spec, self.groups = spec, groups

        def describe(self):
            return {"class": "BEVPriorSubject", "name": "prior", "n_layers": len(self.prob)}

        def predict(self, sample, taps=()):
            dts = {dt: {"prob": self.prob, "dist": self.dist, "mass": None, "dets": [], "detr": None, "vhat": None} for dt in sample.reference}
            return Prediction(outputs={"dt": dts})
    return _BEVSubject, _PriorSubject


def prior_maps(train_ds, n: int, spec):
    """per-cell mean of every evaluated GT layer over n train samples (spread evenly), dt = 0 keyframes"""
    idx = np.linspace(0, len(train_ds) - 1, n).round().astype(int)
    k = layer_index(spec)
    acc, accd = {}, {}
    for i in idx:
        g = train_ds._gt(train_ds.samples[i])["cur"]
        for name, (_, kind, key) in k.items():
            if kind in ("area", "object", "line"):
                acc[name] = acc.get(name, 0) + np.asarray(g[key], np.float64)
            elif kind == "dist":
                accd[name[:-5]] = accd.get(name[:-5], 0) + np.asarray(g[key], np.float64)
    return ({n_: (v / len(idx)).astype(np.float32) for n_, v in acc.items()},
            {n_: (v / len(idx)).astype(np.float32) for n_, v in accd.items()})


# ------------------------------------------------------------------------------------------------ entry
def build(run: str, device: str = "cuda", split: str = "val", workers: int = 2, ckpt_dir: Optional[str] = None, control: str = "",
          prior: int = 0, evaluator: Optional[str] = None):
    """(Source, Subject, Task) of a training run. split: 'val' | 'train' (overfitting check). prior > 0: the Subject is the prior-map
    baseline built from that many train keyframes (no network is loaded). evaluator: the evaluator hash of the Source (default: taken now; the prior
    suite of an `eval` gets the main suite's, so both stores are checked against the same value)."""
    os.chdir(ROOT)
    code = setup_paths(run)
    ns = _chain()
    ns.code = code
    from evaluation.bev_task import BEVTask
    Subject, Prior = _make_subject_classes()
    import torch
    device = device if torch.cuda.is_available() else "cpu"
    with open(os.path.join(run, "config.json")) as f:
        cfg = ns.T.TrainConfig(**json.load(f))
    freeze_scenes(cfg)
    cfg.workers, cfg.seq_len, cfg.batch_size = 0, 1, 1
    if not cfg.radar_cache and os.path.isdir(os.path.join(ROOT, "data", "cache", "radar")):
        cfg.radar_cache = "data/cache/radar"                       # exact (feature_cache radar-selftest): only faster
    train_ds, val_ds = ns.T.make_datasets(cfg)
    ds = train_ds if split == "train" else val_ds
    source = _make_source_class()(cfg, ds, ns, workers=workers, control=control, split=split,
                                      evaluator=evaluator or evaluator_hash())
    if prior:
        model, loss, _ = ns.T.build(cfg)
        spec, grid = model.spec, model.grid
        subject = Prior(prior_maps(train_ds, prior, spec), spec, [g for g, _ in ns.NUSC_GROUPS])
        refine, z_levels = False, tuple(model.cfg.z_levels)
        del model, loss
    else:
        subject = Subject(run, device, ns, ckpt_dir)
        spec, grid = subject.model.spec, subject.model.grid
        refine, z_levels = subject.refine, tuple(subject.model.cfg.z_levels)
    task = BEVTask(grid, spec, [g for g, _ in ns.NUSC_GROUPS], depth_centres=ns.depth_centres, z_levels=z_levels, refine=refine)
    return source, subject, task
