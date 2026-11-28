"""The BEV adapter of the evaluation suite (autoresearch/Evaluation): Source + Subject + Task of a training run `runs/<name>`.

    build(run="runs/t21_ctl_s0", device="cuda", split="val", workers=2, ckpt_dir=None, control="")  ->  (Source, Subject, Task)

    BEVSource      the val (or train) keyframes of the run's config, grouped into scenes (videos) in time order; one frame =
                   the cached-feature batch of the training code (batch of 1) + the GT of -0.5 / 0 / +0.5 s; worker processes read ahead;
                   describe() carries `gt_signature` (the GT identity: GtGenerator is read live) and `evaluator_hash` (the evaluator's
                   source files at build: the store's identity rule refuses a resume with other evaluator code)
    BEVSubject     the run's last checkpoint under its EMA weights, the chain's state carried frame to frame along a scene,
                   forward hooks for the taps
    BEVTask        evaluation/bev_task.py (metrics, views, checks)
The network is net/ (chain.py: the config, the model, the weights, the forward and its loss); nothing here imports training/.
Controls (sanity: the model must lose what the control takes away): control = shuffle_radar | zero_radar | shuffle_cam | zero_cam.
"""
from __future__ import annotations

import glob
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from Evaluation import FrameRef, Prediction, Sample, Source, Subject, TapSpec
from Evaluation.torch_taps import Hook, TapRecorder, count_ops
from bev_fusion.evaluation import bev_sets
from bev_fusion.evaluation.bev_task import BEVTask
from bev_fusion.evaluation.metrics import ms_per_frame
from GtGenerator import version_of_key
from bev_fusion.net import chain, netpath
from bev_fusion.net.model.camera_encoder import FrontCameraEncoder
from bev_fusion.net.model.object_head import decode
from bev_fusion.net.prep.batches import clip_step, make_datasets, to_device
from bev_fusion.net.prep.feature_cache import collate_cached
from bev_fusion.net.prep.geometry import depth_centres
from bev_fusion.net.prep import splits
from bev_fusion.net.prep.nuscenes_dataset import collate
from bev_fusion.net.prep.targets import NUSC_GROUPS, sample_targets

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
AUTORESEARCH = os.path.join(REPO, "autoresearch")
CAMERA_FRAME_KEYS = ("prob", "ctx", "img", "prob2", "ctx2")     # the camera inputs of a frame dict (the camera controls replace all of them)
# the images an item carries beside its frames (lane branch / camera student; lane_img_old: student_frames 'all').
LANE_IMAGE_KEYS = ("lane_img", "lane_img_old")
# A camera control that leaves one in keeps camera information (bev_modules.zero_cam: the same keys)


def perturb_camera(item, other, control):
    """in place, the camera sanity controls on one dataset item: EVERY camera input replaced by `other`'s (shuffle_cam) or zeroed
    (zero_cam): the frames' CAMERA_FRAME_KEYS (both feature caches of camera_features "dual", the image) and the lane branch's key-frame image.
    An unknown camera-like key is refused rather than left in."""
    for f, o in zip(item["frames"], other["frames"]):
        unknown = [k for k in f if k.startswith(("prob", "ctx", "img")) and k not in CAMERA_FRAME_KEYS]
        if unknown:
            raise ValueError(f"camera control: frame keys {unknown} are not handled (add them to CAMERA_FRAME_KEYS)")
        for k in CAMERA_FRAME_KEYS:
            if k in f:
                f[k] = o[k] if control == "shuffle_cam" else f[k] * 0
    unknown = [k for k in item if k.startswith("lane_img") and k not in LANE_IMAGE_KEYS]
    if unknown:
        raise ValueError(f"camera control: item keys {unknown} are not handled (add them to LANE_IMAGE_KEYS)")
    for k in LANE_IMAGE_KEYS:              # the current frame's image and, with model.student_frames 'all', the older frames' images
        if k in item:
            item[k] = other[k] if control == "shuffle_cam" else item[k] * 0


def perturb_radar(item, other, control):
    """in place, the radar sanity controls on one dataset item: the radar of `other` (shuffle_radar) or nothing (zero_radar), for the item's radar set AND, with
    radar_per_frame, for every set of item["radar_seq"] (the radar branch of the recursive unroll reads those, not item["radar"]: a control that leaves them in
    would measure too small a radar dependence). `other` is an item of the same dataset, so it carries radar_seq exactly when `item` does."""
    if control == "shuffle_radar":
        item["radar"] = other["radar"]
        if "radar_seq" in item:
            item["radar_seq"] = other["radar_seq"]
    elif control == "zero_radar":
        item["radar"] = {k: v[:0] for k, v in item["radar"].items()}
        if "radar_seq" in item:
            item["radar_seq"] = [{k: v[:0] for k, v in r.items()} for r in item["radar_seq"]]
    else:
        raise ValueError(f"unknown radar control {control}")


# ------------------------------------------------------------------------------------------------ loading
def load_run(run: str, device: str, ckpt_dir: Optional[str] = None):
    """(cfg, model, loss, step, checkpoint path): the last checkpoint of the run (or of ckpt_dir) under its EMA weights"""
    cfg = chain.read_config(os.path.join(run, "config.json"))
    model, loss, _ = chain.build(cfg)
    model, loss = model.to(device), loss.to(device)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir or run, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir or run}")
    step = chain.load_weights(model, loss, ckpts[-1])
    return cfg, model, loss, step, ckpts[-1]


def gt_version(sig) -> Optional[str]:
    """the GT version key (v13, ...) of a GT identity '<res>_<ss>_<generator signature>'; None when it is not a released version"""
    return version_of_key(sig) if sig else None


# everything the evaluator is made of (the default of evaluator_tree_hash)
EVALUATOR_PATHS = (os.path.join(ROOT, "evaluation"), os.path.join(AUTORESEARCH, "Evaluation"))
# the files that shape the NUMBERS (the resume guard of a store): the metrics, the adapter that predicts, the task and its optional blocks, the Evaluation's
# frame loop, counts, sampling and comparison core, and the views (`bev_views.py` holds `masks()`, every map's denominator, `gt_boxes()` and `layer_index()`;
# `bev_annotation_pts.py` builds the point-count table that the `cov` metric loads; `bev_views.py` also holds the page rendering, so a picture-only edit there
# changes the hash too; `bev_eval.py` is the class interface's adapter: the reference and the metrics without the loss).
# Not the timing, the module probe or the README: an edit there changes no number and must not orphan a store that took 30 minutes
NUMBER_PATHS = tuple(os.path.join(ROOT, "evaluation", f) for f in ("bev_adapter.py", "bev_task.py", "metrics.py", "bev_velocity.py", "bev_fusion_area.py", "bev_curves.py",
                                                                    "bev_views.py", "bev_annotation_pts.py", "bev_eval.py")) + \
    tuple(os.path.join(AUTORESEARCH, "Evaluation", f) for f in ("comparison", "quantitative.py", "store.py", "runner.py", "contracts.py", "sampling.py"))


def evaluator_tree_hash(paths=None) -> str:
    """12 hex over the posix-relative names (sorted) and the bytes, CRLF read as LF, of the python files under `paths` (default: evaluation/ and
    autoresearch/Evaluation/); a path that does not exist is refused (FileNotFoundError)"""
    files = []
    for p in (EVALUATOR_PATHS if paths is None else paths):
        p = os.path.abspath(p)
        if not os.path.exists(p):                                             # a renamed or missing file must not drop out of the hash unseen
            raise FileNotFoundError(f"evaluator path not found: {p}")
        files += [p] if os.path.isfile(p) else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs if f.endswith(".py")]
    h = hashlib.sha256()
    for rel, f in sorted((os.path.relpath(f, REPO).replace("\\", "/"), f) for f in files):
        with open(f, "rb") as fh:
            h.update(rel.encode() + b"\0" + fh.read().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def evaluator_hash() -> Optional[str]:
    """the hash of the files that shape the numbers (NUMBER_PATHS) for the Source's description (None, with a warning, when one of them cannot be read: a
    store started then differs from every other, so a resume is refused rather than mixed)"""
    try:
        return evaluator_tree_hash(paths=NUMBER_PATHS)
    except OSError as ex:
        print(f"WARNING: evaluator hash not available ({type(ex).__name__}: {ex}); the store is not protected against a resume with other evaluator code", flush=True)
        return None


def layer_index(spec):
    return {name: (i, kind, key) for i, (name, kind, key) in enumerate(spec)}


def predictions(p, b: int, spec):
    """one sample's dict of probability maps (areas, lines, objects) and distance fields from the model output p"""
    sem = p["sem"][b].float()
    prob, dist = {}, {}
    for name, (i, kind, key) in layer_index(spec).items():
        if kind in ("area", "object", "line"):
            prob[name] = torch.sigmoid(sem[i]).cpu().numpy()
        elif kind == "dist":
            dist[name[:-len("_dist")]] = sem[i].clamp(min=0).cpu().numpy()
    return prob, dist


def _presence(kind):
    """a 1 m channel that is the presence logit of a polyline layer (layers.layer_spec_1m: kind 'line' with a distance field, 'centreline' without)"""
    return kind in ("line", "centreline")


def predictions_1m(p, b, spec_1m):
    """the 1 m centre-line head of one sample (p["sem_1m"], channels = spec_1m = layers.layer_spec_1m) on the 0.5 m grid of the other maps: ({layer: presence
    probability (X, Y)}, {layer: direction (2, X, Y), the raw (cos, sin)}); every 1 m cell is copied to its 2 x 2 cells (nearest: layers.pool_polyline max-pools
    the 2 x 2 blocks, so block (i, j) holds the cells 2i, 2i + 1, 2j, 2j + 1). The layer is scored under the name of its GT mask (`centreline`)."""
    sem = p["sem_1m"][b].float()
    s = p["sem"].shape[-1] // sem.shape[-1]
    up = lambda a: a.repeat_interleave(s, -2).repeat_interleave(s, -1).cpu().numpy()      # noqa: E731
    idx = {name: i for i, (name, _, _) in enumerate(spec_1m)}
    prob, direction = {}, {}
    for name, (i, kind, _) in layer_index(spec_1m).items():
        if _presence(kind):
            prob[name] = up(torch.sigmoid(sem[i]))
            if name + "_cos" in idx:
                direction[name] = up(sem[[idx[name + "_cos"], idx[name + "_sin"]]])
    return prob, direction


def outputs_of(model, res):
    """the outputs of every dt of one model result (sample 0) the way both Subjects return them (BEVSubject here, bev_free.FreeSubject): maps as probabilities
    and distances, the decoded boxes (`dets`), the velocity maps, and with a 1 m centre-line head its layers and direction"""
    grid, spec = model.grid, model.spec
    dts = {}
    for dt, p in res["out"].items():
        prob, dist = predictions(p, 0, spec)
        dts[dt] = {"prob": prob, "dist": dist,
                   "mass": p["mass"][0, 0].float().cpu().numpy() if p.get("mass") is not None else None,
                   "dets": decode({k: v[:1].float().cpu() for k, v in p["obj"].items()}, grid, NUSC_GROUPS, k=100, threshold=0.05)[0],
                   "vhat": p["v_hat"][0].float().cpu().numpy() if p.get("v_hat") is not None else None,
                   # the object head's raw velocity map (bev_velocity centres)
                   "objvel": p["obj"]["vel"][0].float().cpu().numpy() if "vel" in p["obj"] else None}
        # the 1 m centre-line head: its presence is a layer of `prob`, its direction goes to the angle check
        if model.spec_1m:
            p1, dts[dt]["dir"] = predictions_1m(p, 0, model.spec_1m)
            prob.update(p1)
    return dts


def task_spec(model):
    """the layer rows the Task scores: the model's own (the 0.5 m head) + the presence layers of its 1 m head as line layers of the 0.5 m grid, against the GT
    mask the pooled target comes from (key `centreline@1m` -> `centreline`); a model without the 1 m head: its spec itself. (Not the spec of `predictions`:
    these rows have no channel in p["sem"].)"""
    return model.spec if not model.spec_1m else list(model.spec) + [(n, "line", key.split("@")[0]) for n, kind, key in model.spec_1m if _presence(kind)]


def _identity(x):
    return x


# ------------------------------------------------------------------------------------------------ Source
class BEVSource(Source):
    def __init__(self, cfg, ds, workers: int = 2, control: str = "", split: str = "val", evaluator: Optional[str] = None, set_hash: Optional[str] = None):
        self.cfg, self.ds, self.workers, self.control, self.split = cfg, ds, workers, control, split
        # a named set (small.val): the hash of its scene list, part of the store identity (val / train: None, nothing new)
        self.set_hash = set_hash
        # evaluator_hash(), taken once at build: a resumed store must come from the same evaluator code
        self.evaluator = evaluator
        self.nusc = ds.nusc
        self.fn = collate_cached if cfg.feature_cache else collate
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
        sig = self.ds.gt_signature()
        return {"class": "BEVSource", "dataset": "nuScenes " + self.cfg.version, "split": self.split, "scenes": len(self._frames),
                "frames": len(self.ds), "feature_cache": self.cfg.feature_cache, "control": self.control or None,
                "gt_signature": sig, "gt_version": gt_version(sig), "evaluator_hash": self.evaluator,
                # the scenes this Source really holds (splits.list_hash): a config names them by a split or a list, and a list can change under one name
                "scenes_hash": splits.list_hash(self.scenes()), **({"set_hash": self.set_hash} if self.set_hash else {})}

    # -- read-ahead by worker processes (the Runner announces the frames it will load, in order)
    def prefetch(self, refs: Sequence[FrameRef]) -> None:
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
        scene (the samples are in scene order: a frame a few keyframes later is the same street, which keeps the static layers and makes the control
        weaker than it claims)"""
        c = self.control
        other = self.ds[(self._idx[ref.key] + len(self.ds) // 2) % len(self.ds)]
        if c in ("shuffle_radar", "zero_radar"):
            perturb_radar(item, other, c)
        elif c in ("shuffle_cam", "zero_cam"):
            perturb_camera(item, other, c)
        else:
            raise ValueError(f"unknown control {c}")

    def load(self, ref: FrameRef) -> Sample:
        item = self._item(ref)
        if self.control:
            self._perturb(item, ref)
        batch = self.fn([item])
        gts = sample_targets(batch["gt"])
        nusc, tok = self.nusc, ref.key

        def image():
            sd = nusc.get("sample_data", nusc.get("sample", tok)["data"]["CAM_FRONT"])
            return Image.open(os.path.join(nusc.dataroot, sd["filename"])).convert("RGB")

        def camera():
            return np.asarray(image())

        def camera_model():
            """the image as the network's camera branch sees it: the crop transform of the batch (`ida`), or a plain resize for a batch without one"""
            hw = batch.get("hw", (256, 704))
            H, W = int(hw[0]), int(hw[1])
            if "ida" not in batch:
                return np.asarray(image().resize((W, H)))
            inv = np.linalg.inv(np.asarray(batch["ida"], np.float64))
            return np.asarray(image().transform((W, H), Image.AFFINE, data=tuple(inv[:2].flatten()), resample=Image.BILINEAR))
        return Sample(ref=ref, inputs=batch, reference={dt: gts[dt][0] for dt in gts},
                      display={"camera": camera, "camera_model": camera_model},
                      meta={"timestamp": item.get("timestamp"), "token": tok})


# ------------------------------------------------------------------------------------------------ Subject
def bev_taps(grid):
    ext = (grid.y_min, grid.y_max, grid.x_min, grid.x_max)
    kw = dict(extent=ext, flip_x=True, optional=True)         # which stages exist depends on the configuration
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
    """the taps that are not a module's output, added to `arrays` in place: `hurdle_mov` (the object head's moving logits per class of dt 0) and `radar_aux`
    (the radar auxiliary head, which only runs while training, applied here to the radar branch's output: the radar memory's when the memory is on, the radar
    encoder's otherwise; with the memory on but its tap unreadable the tap is skipped, not fed with the wrong stage)"""
    if "hurdle_mov" in names and "mov" in out0["obj"]:
        arrays["hurdle_mov"] = out0["obj"]["mov"][0].float().cpu().numpy()
    if "radar_aux" in names and getattr(model, "aux_rad", None) is not None:
        feat = arrays.get("radar_memory") if model.rad_temporal is not None else arrays.get("radar_feat")
        if feat is not None:
            arrays["radar_aux"] = model.aux_rad(torch.from_numpy(feat)[None].to(device))[0].float().cpu().numpy()


def _first_of_pair(o):
    """the radar encoder returns (features, lossless state) for radar_kind 'delta_slots' (with_state=True), the features alone otherwise"""
    return o[0] if isinstance(o, tuple) else o


def _state_of_pair(o):
    """the lossless raster of a delta_slots radar encoder; anything else has none (the recorder skips a tap that cannot be read)"""
    return o[1] if isinstance(o, tuple) and len(o) > 1 else None


def bev_hooks() -> dict:
    """the forward hooks of the chain's stages (name -> Evaluation.torch_taps.Hook); recorded only when armed"""
    return {"lift": Hook("lift", pick=lambda o: o[:1], select="first"),
            "cam_reduce": Hook("cam_reduce", pick=lambda o: o[:1], select="first"),
            "cam_temporal": Hook("cam_temporal", pick=lambda o: o[0]),
            "temporal_agg": Hook("cam_temporal", pick=lambda o: o[2]["agg"]),
            "memory_conf": Hook("cam_temporal", pick=lambda o: o[2]["mem_conf"]),
            # first: with radar_per_frame the radar block runs once per camera frame, newest first;
            "radar_raster": Hook("radar.net", when="in", select="first"),
            # the default 'last' would show the OLDEST frame's sweep (a single call: no change)
            "radar_feat": Hook("radar", pick=_first_of_pair, select="first"),
            "radar_state": Hook("radar", pick=_state_of_pair, select="first"),
            "radar_memory": Hook("rad_temporal", pick=lambda o: o[0]),
            "fused": Hook("fusion"),
            "motion_l0": Hook("motion_block", pick=lambda o: o[0]),
            "motion_l1": Hook("motion_block", pick=lambda o: o[1]),
            "motion_l2": Hook("motion_block", pick=lambda o: o[-1])}                    # the finest level, whatever the number of levels


def read_taps(subject, res, batch, names):
    """the tap arrays of one forward of `subject` (a BEVSubject): the armed recorder's, then the ones read from the result (trunk, the output taps, the camera
    visibility)"""
    model, device = subject.model, subject.device
    arrays = subject.rec.read(0) if names else {}
    if "memory_conf" in arrays:
        arrays["memory_conf"] = arrays["memory_conf"][0]
    if "trunk" in names:
        arrays["trunk"] = res["features"][0].float().cpu().numpy()
    output_taps(model, res["out"][0.0], arrays, names, device)
    if "camera_visibility" in names:
        arrays["camera_visibility"] = model.camera_visibility(to_device(batch, device), res["features"])[0, 0].float().cpu().numpy()
    return {k: v for k, v in arrays.items() if k in names}


class BEVSubject(Subject):
    def __init__(self, run, device, ckpt_dir=None):
        self.run = run
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cfg, self.model, self.loss, self.step, self.ckpt = load_run(run, self.device, ckpt_dir)
        self.warmup = max(0, int(self.cfg.seq_len) - 1)
        self.rec = TapRecorder(self.model, bev_hooks())
        self.state, self.prev = None, None

    def reset(self):
        self.state, self.prev = None, None

    def taps(self):
        return bev_taps(self.model.grid)

    def describe(self):
        return {"class": "BEVSubject", "name": os.path.basename(os.path.normpath(self.run)), "run": self.run,
                "checkpoint": os.path.basename(self.ckpt), "step": self.step, "weights": "EMA", "device": self.device,
                "seq_len": self.cfg.seq_len}

    def ops(self, sample):
        """GMACs of the cached chain at batch 1 with dts -0.5 / 0 / +0.5 (the path `benchmark` times), counted by torch; the pre-trained
        camera heads are a fixed cost that is not in it"""
        b1 = to_device(sample.inputs, self.device)
        r = count_ops(lambda: self.model(b1, dts=(-0.5, 0.0, 0.5)))
        r["note"] += "; the cached chain at batch 1 with dts -0.5 / 0 / +0.5 (what benchmark() times), without the pre-trained camera heads"
        return r

    def arm(self, taps):
        """arm the recorder for the taps asked for -> their names; the radar aux head reads the radar branch, so its input is recorded too (not returned)"""
        names = set(taps)
        self.rec.arm([n for n in (names | ({"radar_feat", "radar_memory"} if "radar_aux" in names else set())) if n in self.rec.hooks])
        return names

    def carried(self):
        """the forward arguments that carry the memory: the state of the previous frame of the scene and the ego step since it; {} at the first frame"""
        return {} if self.state is None else {"state": self.state, "step": clip_step(self.prev, self.device)}

    def predict(self, sample, taps=()):
        names = self.arm(taps)
        batch = sample.inputs
        with torch.no_grad():
            out, res, _ = chain.forward_loss(self.model, self.loss, batch, 1.0, self.device, self.cfg, train_terms=False, **self.carried())
            self.state, self.prev = res["state"], batch
            dts = outputs_of(self.model, res)
            arrays = read_taps(self, res, batch, names)
        loss = {k: float(v) for k, v in out.items() if k == "total" or k.startswith("group.")}
        return Prediction(outputs={"dt": dts}, taps=arrays, info={"loss": loss})

    def benchmark(self, sample, n=30, warmup=5):
        """ms per frame of the CACHED chain at batch 1 (the pre-trained camera heads are a fixed cost, timed separately):
        dt 0 and (-0.5, 0, +0.5); only valid on an idle GPU (the load is recorded by the caller)"""
        b1 = to_device(sample.inputs, self.device)
        out = {}
        with torch.no_grad():
            out["bev_dt0_median"] = ms_per_frame(lambda: self.model(b1, dts=(0.0,)), n=n, warmup=warmup, batch1=False)["median"]
            m3 = ms_per_frame(lambda: self.model(b1, dts=(-0.5, 0.0, 0.5)), n=n, warmup=warmup, batch1=False)
            out["bev_3dt_median"] = m3["median"]
            cam = FrontCameraEncoder().to(self.device).eval()
            x, cv = torch.randn(1, 3, 256, 704, device=self.device), torch.zeros(1, 27, device=self.device)
            out["camera_heads_median"] = ms_per_frame(lambda: cam.image_features(x, cv), n=min(n, 30), warmup=5, batch1=False)["median"]
            del cam
        out["median_ms"], out["p90_ms"], out["mean_ms"], out["n"] = m3["median"], m3["p90"], m3["mean"], n
        return out

    def close(self):
        self.rec.close()


# ------------------------------------------------------------------------------------------------ entry
def build(run: str, device: str = "cuda", split: str = "val", workers: int = 2, ckpt_dir: Optional[str] = None, control: str = "", evaluator: Optional[str] = None):
    """(Source, Subject, Task) of a training run. split: the set to evaluate on (evaluation/bev_sets.py): 'val' | 'train' = the run's own lists (the train one
    is the overfitting check), '<split>.<part>' (small.val) = the lists of that file of splits/ put into the config before the data is made. evaluator: the
    evaluator hash of the Source (default: taken now). Works in the problem's data folder (netpath.SITE): the configs name their files relative to it."""
    run = os.path.abspath(run)
    os.chdir(netpath.SITE)
    device = device if torch.cuda.is_available() else "cpu"
    cfg = chain.read_config(os.path.join(run, "config.json"))
    # the run's own lists for val / train, the file's for small.val
    cfg.train_scenes, cfg.val_scenes = bev_sets.lists(split, cfg.train_scenes, cfg.val_scenes)
    if not cfg.radar_cache and os.path.isdir(os.path.join("data", "cache", "radar")):
        cfg.radar_cache = "data/cache/radar"                       # exact: only faster
    train_ds, val_ds = make_datasets(cfg)
    ds = train_ds if bev_sets.part_of(split) == "train" else val_ds
    named = bev_sets.parse(split)[0] is not None
    source = BEVSource(cfg, ds, workers=workers, control=control, split=split, evaluator=evaluator or evaluator_hash(), set_hash=bev_sets.set_hash(split) if named else None)
    # make_datasets drops scenes that are not in the official part or have no front-camera files, silently: not for a set
    if named:
        want = bev_sets.lists(split, [], [])[0 if bev_sets.part_of(split) == "train" else 1]
        gone = sorted(set(want) - set(source.scenes()))
        if gone:
            raise SystemExit(f"set {split}: {len(gone)} of its {len(want)} scenes cannot be read by this run's data ({', '.join(gone[:5])}{', ...' if len(gone) > 5 else ''}): "
                             f"not in the official {bev_sets.part_of(split)} split of nuScenes, or no front-camera files; its features / GT caches may still be missing too")
    subject = BEVSubject(run, device, ckpt_dir)
    task = BEVTask(subject.model.grid, task_spec(subject.model), [g for g, _ in NUSC_GROUPS], depth_centres=lambda: depth_centres().numpy(),
                   z_levels=tuple(subject.model.cfg.z_levels))
    return source, subject, task
