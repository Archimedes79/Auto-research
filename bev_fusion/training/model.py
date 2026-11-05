"""The whole camera + radar BEV chain, assembled from the blocks of training/ (plan.md sections 2-8).

    front camera frames (current + K-1 past, 12 Hz)            radar detections of all sweeps of the last 0.5 s
        |  pre-trained DualBEV image heads (frozen)                  |  already at the camera time (exact poses +
        |  -> depth distribution + context per pixel group           |     radial Doppler motion; nuscenes_dataset.py)
        |  lift (PD / P / S) per frame -> 1x1 reduce                  |  radar block (raster + properties + 2 convs)
        |  camera temporal fusion (window of original frames         |  radar temporal fusion (optional)
        |     + world-aligned memory / ego-warp GRU / none)          |
        +---------------------------- fusion (concat / gate / CMX) --+
                                         |
                           BEV trunk (ResNet pyramid + FPN)
                                         |
            shared TimeMover (velocity head + splat to t + dt) -> TimedHead(semantic) / TimedHead(objects)
            auxiliary heads on each branch (training only)

The five layers (training/architecture.md): precompute (precompute.py: the visibility maps, made at the entry of forward only where a
memory reads them), input interface (interface.py), the network slots below (one builder table per slot, check_config), output
interface (per dt the maps and object maps; with dense_outputs the dense boxes and the soft line map of extract.py), post-processing
(boxes(): extract.boxes_from_maps, the one box extractor; object_head.decode is the same function).
Every block is chosen in ModelConfig; each option is either the plan's design or its paper fallback:
  lift          "pd"  lift PD, pull with depth weight (DualBEV / FastBEV++ / FB-BEV) - default
                "p"   pull without depth (Simple-BEV, Fast-BEV) - control
                "s"   forward splat (LSS / BEVPoolV2) - training/fallbacks.py LiftS
  cam_temporal  "window_memory"  training/temporal_fusion.py TemporalFusion (window of original frames + world memory)
                "ego_gru"        FIERY-style ego warp + ConvGRU - training/fallbacks.py EgoWarpGRU (control arm A)
                "none"           current frame only
  cam_buffer    "window" (default) the batch holds the window of n_cam frames, all lifted on every call; a static world memory is the only
                state | "pair" the window is ONE frame (t-1): the batch holds 2 frames, or in stream mode ONE frame and the previous raw
                camera BEV comes from state["buf"] | "recurrent" the temporal block's own output of the previous frame is the buffer
                (state["buf"] = raw + fused): each frame is lifted once and fused with the previous fused state (splat_window on it, its
                velocity head); a batch unrolls its F frames oldest -> newest, starting from the previous call's buffer (training: the
                previous keyframe; none: cold). Train / deploy: training chains up to 3 keyframes x F frames, a stream is unbounded. The motion
                block reads two RAW BEVs (current + the frame before it) in both modes. The radar keeps its own 0.5 s point buffer
  camera_student "" (default) | "evit_b1" | "resnet18" (t22 stage B, training/student_encoder.py): a small trained image encoder replaces the cached teacher outputs of the
                current frame (both feature sources of camera_features 'dual'); older frames keep their cached outputs
  rad_temporal  "none" (default: the 0.5 s of sweeps are already accumulated at point level), "window_memory", "ego_gru"
                (rad_static_gate: the memory writes only the cells the measured Doppler calls static)
  fusion        "concat" (BEVFusion ConvFuser) - default, "gate" (GaussianCaR FeatureRectify), "cmx" (GaussianCaR's
                4-stage fuser + decoder, training/fallbacks.py CMXTrunk: replaces fusion AND backbone)
  heads         semantic (channels = losses.layer_spec, made from the layer table layers.py; centreline_1m adds the 1 m branch sem_1m) and
                objects (CenterPoint, S5a), both decoded at any dt by
                training/timed_head.py (the shared TimeMover moves S to dt); the output is the prediction layout of losses.BEVLoss;
                S5c refinement works on the decoded boxes (object_head.py). forward(ego=...) takes the true gap per sample.
Conventions as the blocks: grid x 0..60 m forward, y -26..26 m left, 0.5 m; maps (B, C, X, Y).
Run the self-test (nuScenes mini, 2 samples): python training/model.py
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import launch_saving as LS
    from .augment import flip_cam
    from .bev_trunk import BEVTrunk, SemanticHead
    from .blocks_new import CtxAdapter, RadarPillars, SecondTrunk, ZSumReduce
    from .camera_encoder import DEPTH_BINS, FrontCameraEncoder, depth_centres
    from .extract import boxes_from_maps, dense_boxes, soft_line_map
    from .fusion import ConcatFusion, GateFusion
    from .lane_branch import LaneBranch
    from .lift_pd import Grid, LiftPD, lift_key
    from .layers import LEVELS
    from .losses import NUSC_GROUPS, layer_spec, layer_spec_1m, radar_aux_spec, refine_gt_boxes, sample_targets
    from .object_head import BoxRefiner, ObjectHead, decode, refine_boxes, refine_candidates
    from .precompute import Precompute
    from .radar_bev import RadarBEV
    from .radar_encoder import RadarEncoder, n_channels
    from .temporal_fusion import TemporalFusion, as_chain
    from .motion_block import MotionBlock, block_inputs
    from .timed_head import TimedHead, TimeMover
except ImportError:                                   # run as a script
    import launch_saving as LS
    from augment import flip_cam
    from bev_trunk import BEVTrunk, SemanticHead
    from blocks_new import CtxAdapter, RadarPillars, SecondTrunk, ZSumReduce
    from camera_encoder import DEPTH_BINS, FrontCameraEncoder, depth_centres
    from extract import boxes_from_maps, dense_boxes, soft_line_map
    from fusion import ConcatFusion, GateFusion
    from lane_branch import LaneBranch
    from lift_pd import Grid, LiftPD, lift_key
    from layers import LEVELS
    from losses import NUSC_GROUPS, layer_spec, layer_spec_1m, radar_aux_spec, refine_gt_boxes, sample_targets
    from object_head import BoxRefiner, ObjectHead, decode, refine_boxes, refine_candidates
    from precompute import Precompute
    from radar_bev import RadarBEV
    from radar_encoder import RadarEncoder, n_channels
    from temporal_fusion import TemporalFusion, as_chain
    from motion_block import MotionBlock, block_inputs
    from timed_head import TimedHead, TimeMover

Z_LEVELS = (-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)        # a slice at the road plane (oracle-lift gate fix)


def _fallbacks():
    try:
        from . import fallbacks
    except ImportError:
        import fallbacks
    return fallbacks


@dataclass
class ModelConfig:
    camera_features: str = "dualbev"   # "dualbev" (frozen DualBEV heads, stride 16) | "bevfusion" (cached BEVFusion features, stride 8; needs feature_cache) | "dual" (both, needs feature_cache + feature_cache2)
    lift: str = "pd"
    cam_temporal: str = "window_memory"
    cam_buffer: str = "window"         # t21: what the camera temporal block carries from the past: "window" (the batch's n_cam frames, today) | "pair" (the
                                       # previous raw BEV) | "recurrent" (the previous fused output); the last two need cam_temporal 'window_memory'
    rad_temporal: str = "none"
    rad_static_gate: bool = False      # the radar memory writes only where the measured Doppler says static (precompute.radar_static), not
                                       # everywhere a detection is (radar_hits): a moving car is not remembered; needs rad_temporal
    rad_static_vr: float = 1.0         # the Doppler (m/s) at which a detection counts as 1/e static: weight exp(-(vr / rad_static_vr)^2)
    fusion: str = "concat"
    c_cam: int = 64
    c_rad: int = 32
    c_fuse: int = 96
    trunk_width: float = 0.5
    trunk_detail: int = 0              # stride-1 detail path (bev_trunk.DetailPath) channels, 0 = off
    head_channels: int = 0             # 1x1 reduction of the trunk output before the heads (CenterPoint-style light heads), 0 = off
    trunk_pretrained: str = ""         # DualBEV checkpoint to initialise the BEV encoder (width 1.0, c_fuse 80), "" = random
    trunk_kind: str = "dualbev"        # "dualbev" (ResNet pyramid + FPN-LSS) | "second" (SECOND + SECONDFPN, blocks_new.py)
    trunk_fpn_original: bool = False   # t4: FPN-LSS concatenation order of BEVDet / DualBEV (stage 0 first), what the pre-trained weights expect
    radar_kind: str = "raster"         # "raster" (means + 2 convs) | "pillars" (point MLP + scatter-max, blocks_new.py) | "off" (zeros)
                                       # | "delta_slots" (radar_encoder.RadarEncoder: lossless delta-coded slot raster + 2 convs; needs c_rad >= 61;
                                       # its reconstruction side loss is switched on by TrainConfig.radar_aux)
    cam_collapse: str = "zconv"        # try 7: "zconv" (1x1 conv over Z x 80 channels) | "zsum" (sum over heights, then 1x1; blocks_new.ZSumReduce)
    cam_adapter: bool = False          # try 6 (t3_improve_loop): trainable residual adapter on the image context features before the lift
    camera_off: bool = False           # study arm: the camera BEV is zero (radar-only system), the image heads are not run
    p_drop: float = 0.2                # modality dropout of the fusion in training (fusion.drop_modality: per sample the camera or the
                                       # radar input is zeroed, p / 2 each); forced to 0 when a sensor is off (camera_off, radar_kind
                                       # "off"): dropping the only live sensor would blank the sample (review 2026-10-31 pass 2)
    lane_branch: str = ""              # t7: stride-4 lane branch of the current frame added to the camera BEV after the temporal
                                       # block (lane_branch.py): "" = off | "res4" (frozen ResNet-50 layer 1) | "rgb" (CNN from scratch);
                                       # camera_features 'dualbev' or 'dual' (t20: it reads the first source = DualBEV, and adds to the merged BEV)
    lane_pretrained: str = "models/pretrained/dualbev-r50-cbgs.pth"    # weights of the "res4" features
    camera_student: str = ""           # t22 stage B: "" = off | a key of STUDENTS (training/student_encoder.py): a small trained image encoder that replaces the cached teacher outputs of
                                       # the CURRENT frame (frame 0) of both feature sources, from batch["lane_img"] (TrainConfig.lane_images); older frames keep their cached teacher
                                       # outputs. Needs the cached chain with camera_features 'dual'; module `student`, built LAST (every other tensor's init is the same on or off)
    student_init: str = ""             # a distillation run folder (its newest checkpoint) or a checkpoint file: the student starts from its EMA weights; "" = random (only with student_trainable)
    student_trainable: bool = False    # True: the student trains in the chain (lr group 'student', BatchNorm statistics fixed); False: frozen, not part of the checkpoints (loaded from student_init)
    student_frames: str = "current"    # the frames the student replaces: "current" (frame 0, the only frame with a JPEG) | "all" (needs a sweep-image cache: NotImplementedError)
    oversample: int = 4
    centrelines: bool = False          # head layout = losses.layer_spec(centrelines, dense_objects)
    centreline_1m: bool = False        # plan 2026-11-03: a 1 m branch (SemanticHead on the sem head's state, 2 x 2 average-pooled) with the centre-line row of the layer
                                       # table (layers.py): output p["sem_1m"] = losses.layer_spec_1m; LossConfig.centreline_1m must agree. Off = no new parameter
    dense_objects: bool = True
    radar_stationary: bool = True      # radar aux head = losses.radar_aux_spec(radar_stationary)
    n_sem: int = None                  # optional check: must equal len(layer_spec(...)) when given (train.py passes it)
    aux_heads: bool = True
    object_direction: bool = False     # object head: sign-free axis + front / back class (object_head.ObjectHead)
    object_moving: bool = False        # object head: moving / static logit, velocity on moving centres only (t11 hurdle)
    motion_block: bool = False         # t15: the time mover's velocity field from training/motion_block.py (the current and the oldest
                                       # camera frame - with cam_buffer 'pair' / 'recurrent' the frame just before the current one - + the radar
                                       # Doppler, a 3-level pyramid) instead of the 1x1 head on the trunk; needs >= 2 camera frames (n_cam) and a radar
    motion_strides: tuple = (4, 2, 1)  # its pyramid levels in cells, coarse to fine; (4, 2) = the runtime arm without the 0.5 m level
    box_vel_from_field: bool = False   # t15: the object head's `vel` = the motion block's field (at dt != 0 moved with the state, the
                                       # vector turned with the ego yaw); the object head's own vel branch stays (unused for the output)
    share_mover: bool = False          # D53: the semantic and the object head read the same moved state: ONE splat per dt instead of two (same math; the
                                       # sum of the two heads' gradients meets in one backward node, so a GPU run is equal up to rounding, a CPU forward bit for bit)
    splat_fused: bool = False          # D53: splat.splat scatters the four bilinear corners in one call (launch_saving.py: same arithmetic, fewer launches)
    sync_free: bool = False            # D53 / D54: fixed-shape / host-side forms of the data-dependent steps of the forward (radar raster, memory re-centring; the dt embedding filled on the GPU) and the reads of batch["host"] (projection-table cache key, flip flags) instead of waits for the GPU; launch_saving.py
    refine: bool = False               # S5c: deformable box refinement (object_head.BoxRefiner, n = 1, dt-aware) on the
                                       # object head's features at the target time; trained on decoded + jittered GT
    refine_k: int = 50                 # candidates: top-k decoded boxes per sample + refine_jitter copies of each GT box
    refine_jitter: int = 1
    pretrained: str = "models/pretrained/dualbev-r50-cbgs.pth"
    freeze_camera: bool = True
    z_levels: tuple = Z_LEVELS
    grid_kwargs: dict = field(default_factory=dict)
    dense_outputs: bool = False        # output interface: per dt the dense box output and the soft line map (extract.py); on for the
                                       # planner and the prototype, off in training and timing (no loss reads them; stored ms stay comparable)


# ------------------------------------------------------------------ the network slots: one builder table each (kind -> block)
def _bevfusion_depths():
    try:
        from .bevfusion_camera import DEPTHS
    except ImportError:
        from bevfusion_camera import DEPTHS
    return DEPTHS


def _zconv(nz, n_ctx, c):
    return nn.Sequential(nn.Conv2d(nz * n_ctx, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))


def _dualbev_trunk(cfg, p_drop):
    trunk = BEVTrunk(cin=cfg.c_fuse, width=cfg.trunk_width, detail=cfg.trunk_detail, fpn_original_order=cfg.trunk_fpn_original)
    if cfg.trunk_pretrained:                                          # detection-trained BEV encoder of DualBEV (SOA init)
        rep = trunk.load_pretrained(cfg.trunk_pretrained)
        if rep["missing"] or rep["shape_mismatch"]:
            raise RuntimeError(f"trunk weights do not fit: {rep['missing'][:3]} {rep['shape_mismatch'][:3]}")
    return trunk, trunk.img_bev_encoder_neck.out_channels + cfg.trunk_detail


def _second_trunk(cfg, p_drop):
    trunk = SecondTrunk(cin=cfg.c_fuse)
    return trunk, trunk.out_channels


def _cmx_trunk(cfg, p_drop):                                          # CMX: its own pyramid + decoder, replaces fusion AND backbone
    return _fallbacks().CMXTrunk(cfg.c_cam, cfg.c_rad, embed=cfg.c_fuse, c_out=128, p_drop=p_drop), 128


@dataclass(frozen=True)
class CameraSource:
    """One frozen image encoder whose (cached) outputs feed the camera slot: the feature family fixes the depth bins, the cell
    convention and the cache's feature size; the suffix names its batch fields (prob<suffix>, ctx<suffix>, ida<suffix>) and, after the
    first source, its modules (lift<suffix>, cam_reduce<suffix>: the checkpoint keys of the t10 chain)."""
    features: str      # "dualbev" (stride 16, cell centres, depth bins 1.25 + 0.5 i) | "bevfusion" (stride 8, frustum cells, bins 1.0 + 0.5 i)
    cache: str         # TrainConfig field holding the cache folder
    suffix: str        # "" for the first source, "2" for the second


# camera slot: kind -> its feature sources, one lift + height collapse each; two or more are merged (Conv 1x1 over the concatenated BEVs).
# The first source is the geometry reference (visibility, precompute); the dataset, collate_cached, the cached model, train.build and
# the tests read this table, not the kind's name.
CAMERAS = {"dualbev": (CameraSource("dualbev", "feature_cache", ""),),
           "bevfusion": (CameraSource("bevfusion", "feature_cache", ""),),
           "dual": (CameraSource("dualbev", "feature_cache", ""), CameraSource("bevfusion", "feature_cache2", "2"))}
CAMERA_FEATURES = tuple(CAMERAS)


def source_lift(src, cfg, grid):
    """the lift of one source: its depth bins and its cell convention (frustum cells for the BEVFusion family)"""
    if src.features == "bevfusion":
        return LIFTS[cfg.lift](grid, _bevfusion_depths(), cfg, frustum_cells=True)
    return LIFTS[cfg.lift](grid, depth_centres(), cfg)


LIFTS = {"pd": lambda grid, depths, cfg, **kw: LiftPD(grid, use_depth=True, oversample=cfg.oversample, depth_centres=depths, **kw),
         "p": lambda grid, depths, cfg, **kw: LiftPD(grid, use_depth=False, oversample=cfg.oversample, depth_centres=depths, **kw),
         "s": lambda grid, depths, cfg, **kw: _fallbacks().LiftS(grid, depth_centres())}
COLLAPSES = {"zconv": _zconv, "zsum": ZSumReduce}                     # height collapse of the lifted volume
TEMPORALS = {"window_memory": lambda c, grid: TemporalFusion(c, c, grid),     # temporal slot, the same table for camera and radar
             "ego_gru": lambda c, grid: _fallbacks().EgoWarpGRU(c, c, grid),
             "none": lambda c, grid: None}
RADARS = {"raster": lambda grid, c: RadarBEV(grid, c), "off": lambda grid, c: RadarBEV(grid, c),
          "pillars": lambda grid, c: RadarPillars(grid, c),
          "delta_slots": lambda grid, c: RadarEncoder(grid, C=c)}      # radar_encoder.py: delta-coded slot raster, pass-through start, side loss
CAM_BUFFERS = ("window", "pair", "recurrent")                          # what the camera temporal block gets from the past (forward: _camera_buffered)
FUSIONS = {"concat": ConcatFusion, "gate": GateFusion, "cmx": None}  # cmx: the backbone takes both branch BEVs
TRUNKS = {"dualbev": _dualbev_trunk, "second": _second_trunk}
BACKBONES = dict(TRUNKS, cmx=_cmx_trunk)


def _student(backbone):
    try:
        from .student_encoder import StudentEncoder
    except ImportError:
        from student_encoder import StudentEncoder
    return StudentEncoder(backbone, pretrained=False, warp_h8=True)           # the weights come from student_init (or from the training)


# camera student (t22 stage B, option camera_student): kind -> the encoder that stands in for the cached teachers of the current frame; which of its outputs replaces which
# feature family's cached tensors: StudentEncoder.HEAD_OF
STUDENTS = {"evit_b1": lambda: _student("efficientvit_b1"), "resnet18": lambda: _student("resnet18")}


def _build_student(cfg):
    """the module `student` of ModelConfig.camera_student: its trained weights (student_init), BatchNorm statistics fixed, frozen unless student_trainable"""
    student = STUDENTS[cfg.camera_student]()
    if cfg.student_init:
        try:
            from .student_encoder import load_trained
        except ImportError:
            from student_encoder import load_trained
        load_trained(student, cfg.student_init)
    student.keep_bn_eval()
    student.requires_grad_(cfg.student_trainable)
    return student


def check_config(cfg):
    """every unknown kind and every combination of slot kinds that does not fit, in one place (one ValueError naming all)"""
    errs = [f"{name}={val!r}: one of {sorted(table)}" for name, val, table in
            (("camera_features", cfg.camera_features, CAMERAS), ("lift", cfg.lift, LIFTS), ("cam_collapse", cfg.cam_collapse, COLLAPSES),
             ("cam_temporal", cfg.cam_temporal, TEMPORALS), ("cam_buffer", cfg.cam_buffer, CAM_BUFFERS), ("rad_temporal", cfg.rad_temporal, TEMPORALS),
             ("radar_kind", cfg.radar_kind, RADARS), ("fusion", cfg.fusion, FUSIONS), ("trunk_kind", cfg.trunk_kind, TRUNKS))
            if val not in table]
    if len(CAMERAS.get(cfg.camera_features, ())) > 1 and (cfg.lift == "s" or cfg.cam_adapter):
        errs.append(f"camera_features={cfg.camera_features!r} (several feature sources) is not supported with lift 's' or cam_adapter")
    if cfg.camera_features == "bevfusion" and cfg.lift == "s":
        errs.append("camera_features='bevfusion' is not supported with lift 's'")
    if cfg.lane_branch and (cfg.camera_features not in CAMERAS or CAMERAS[cfg.camera_features][0].features != "dualbev" or cfg.lift == "s"):
        errs.append("lane_branch reads the depth, the calibration and the crop of the FIRST feature source (the lane image is the DualBEV one): it needs "
                    "camera_features 'dualbev' or 'dual' and lift 'pd' / 'p'")
    if cfg.cam_buffer != "window" and cfg.cam_temporal != "window_memory":
        errs.append(f"cam_buffer={cfg.cam_buffer!r} buffers a frame for the window-memory block: it needs cam_temporal 'window_memory' (is {cfg.cam_temporal!r})")
    if cfg.radar_kind == "delta_slots" and cfg.c_rad < n_channels():
        errs.append(f"radar_kind='delta_slots' needs c_rad >= {n_channels()} (one channel per input channel: the pass-through start); c_rad = {cfg.c_rad}")
    if cfg.rad_static_gate and (cfg.rad_temporal == "none" or cfg.radar_kind == "off"):
        errs.append("rad_static_gate gates the write of the radar memory: it needs rad_temporal 'window_memory' / 'ego_gru' and a radar")
    if cfg.rad_static_vr <= 0:
        errs.append(f"rad_static_vr must be positive (m/s), is {cfg.rad_static_vr}")
    if cfg.lane_branch and cfg.camera_off:                            # the branch reads the current image: a "camera off" arm would still see it
        errs.append("lane_branch reads the camera image: it cannot be combined with camera_off")
    if cfg.camera_student:                                            # t22 stage B
        if cfg.camera_student not in STUDENTS:
            errs.append(f"camera_student={cfg.camera_student!r}: one of {sorted(STUDENTS)}")
        if cfg.camera_features != "dual":
            errs.append("camera_student replaces the cached outputs of BOTH feature sources (DualBEV and BEVFusion) of the current frame: it needs camera_features 'dual'")
        # lift 's' and cam_adapter need no rule of their own: 'dual' (several feature sources) refuses both above
        if cfg.camera_off:
            errs.append("camera_student reads the camera image: it cannot be combined with camera_off")
        if not cfg.student_trainable and not cfg.student_init:
            errs.append("a frozen student (student_trainable False) needs student_init: a random frozen encoder stands in for nothing")
    elif cfg.student_init or cfg.student_trainable:
        errs.append("student_init / student_trainable do nothing without camera_student")
    if cfg.student_frames not in ("current", "all"):
        errs.append(f"student_frames={cfg.student_frames!r}: one of ['all', 'current']")
    if cfg.motion_block and (cfg.camera_off or cfg.radar_kind == "off"):
        errs.append("motion_block reads two camera frames and the radar Doppler: it needs the camera and a radar")
    if cfg.box_vel_from_field and not cfg.motion_block:
        errs.append("box_vel_from_field reads the motion block's field: it needs motion_block")
    if cfg.motion_block and (not cfg.motion_strides or list(cfg.motion_strides) != sorted(cfg.motion_strides, reverse=True)):
        errs.append(f"motion_strides must be a non-empty decreasing tuple of cell strides, is {cfg.motion_strides}")
    elif cfg.motion_block:
        g = Grid(z_levels=cfg.z_levels, **cfg.grid_kwargs)
        if any(g.nx % s or g.ny % s for s in cfg.motion_strides):
            errs.append(f"motion_strides {cfg.motion_strides} must divide the grid {g.nx} x {g.ny}")
    if cfg.centreline_1m:
        g = Grid(z_levels=cfg.z_levels, **cfg.grid_kwargs)
        if g.nx % LEVELS["1m"] or g.ny % LEVELS["1m"]:
            errs.append(f"centreline_1m pools the grid by {LEVELS['1m']}: {g.nx} x {g.ny} must be even")
    if errs:
        raise ValueError("; ".join(errs))
    if cfg.student_frames == "all":
        raise NotImplementedError("student_frames='all' would run the student on the older frames of the window / pair too: they are sweeps, and no sweep-image cache exists "
                                  "(about 58 k JPEGs, 3 GB, a separate step); only 'current' is supported")


class BEVFusionModel(nn.Module):
    CACHED = False                         # True in the cached model (feature_cache.CachedBEVFusionModel): camera_student replaces cached outputs, so it needs it
    _student_side = None                   # camera_student: the student's and the cached teacher's current-frame tensors of the last camera_bevs call, taken by _forward

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        check_config(cfg)
        if cfg.camera_student and not self.CACHED:
            raise ValueError("camera_student replaces cached teacher outputs: it needs the cached chain (a feature_cache; feature_cache.CachedBEVFusionModel), not the live camera heads")
        self.grid = Grid(z_levels=cfg.z_levels, **cfg.grid_kwargs)
        # camera slot: pre-trained image heads (only image_features is used; the cached chain drops them) + lift + height collapse
        self.camera = FrontCameraEncoder()
        if cfg.pretrained:
            rep = self.camera.load_pretrained(cfg.pretrained)
            if rep["missing"] or rep["shape_mismatch"]:
                raise RuntimeError(f"camera weights do not fit: {rep}")
        if cfg.freeze_camera:
            self.camera.freeze_camera_heads()
        del self.camera.lift, self.camera.out_conv                   # replaced by the model's own lift
        n_ctx, nz = 80, len(cfg.z_levels)
        srcs = CAMERAS[cfg.camera_features]                          # one lift + collapse per feature source (the order = the checkpoint's)
        self.lift = source_lift(srcs[0], cfg, self.grid)
        self.cam_adapter = CtxAdapter(n_ctx) if cfg.cam_adapter else None
        self.cam_reduce = COLLAPSES[cfg.cam_collapse](nz, n_ctx, cfg.c_cam)
        for src in srcs[1:]:                                         # t10 "dual": lift2 + cam_reduce2 (BEVFusion features, stride 8)
            setattr(self, "lift" + src.suffix, source_lift(src, cfg, self.grid))
            setattr(self, "cam_reduce" + src.suffix, COLLAPSES[cfg.cam_collapse](nz, n_ctx, cfg.c_cam))
        if len(srcs) > 1:                                            # the sources' BEVs merged back to c_cam
            self.cam_merge = nn.Sequential(nn.Conv2d(len(srcs) * cfg.c_cam, cfg.c_cam, 1, bias=False), nn.BatchNorm2d(cfg.c_cam),
                                           nn.ReLU(True))
        # temporal slot, one block per branch (the same table)
        self.cam_temporal = TEMPORALS[cfg.cam_temporal](cfg.c_cam, self.grid)
        # lane add-on (option): stride-4 branch of the current frame, added after the camera's temporal block
        self.lane = (LaneBranch(self.grid, depth_centres(), kind=cfg.lane_branch, c_out=cfg.c_cam, oversample=cfg.oversample,
                                pretrained=cfg.lane_pretrained) if cfg.lane_branch else None)
        # radar slot
        self.radar = RADARS[cfg.radar_kind](self.grid, cfg.c_rad)
        self.rad_temporal = TEMPORALS[cfg.rad_temporal](cfg.c_rad, self.grid)
        self._rad_mask = "radar_static" if cfg.rad_static_gate else "radar_hits"        # the write mask of the radar memory (precompute.MAPS)
        # fusion + backbone slots (CMX is its own four-stage pyramid + decoder: it replaces both)
        p_drop = 0.0 if (cfg.camera_off or cfg.radar_kind == "off") else cfg.p_drop
        self.fusion = None if cfg.fusion == "cmx" else FUSIONS[cfg.fusion](cfg.c_cam, cfg.c_rad, cfg.c_fuse, p_drop=p_drop)
        self.trunk, c = BACKBONES["cmx" if cfg.fusion == "cmx" else cfg.trunk_kind](cfg, p_drop)
        if hasattr(self.trunk, "head"):
            del self.trunk.head                                      # the heads below decode the trunk features
        self.head_reduce = None
        if cfg.head_channels and cfg.head_channels != c:
            self.head_reduce = nn.Sequential(nn.Conv2d(c, cfg.head_channels, 1, bias=False),
                                             nn.BatchNorm2d(cfg.head_channels), nn.ReLU(True))
            c = cfg.head_channels
        self.spec = layer_spec(cfg.centrelines, cfg.dense_objects)
        if cfg.n_sem is not None and cfg.n_sem != len(self.spec):
            raise ValueError(f"n_sem {cfg.n_sem} != len(layer_spec) {len(self.spec)}")
        self.mover = TimeMover(c, self.grid)
        self.motion_block = None
        if cfg.motion_block:                                     # t15: the field comes from the block; the 1x1 head is removed (built by
            self.motion_block = MotionBlock(cfg.c_cam, self.grid, strides=tuple(cfg.motion_strides))   # TimeMover, then dropped: it would
            self.mover.motion = None                             # get no gradient; TimeMover then needs v_field on every call)
        self.sem_head = TimedHead(SemanticHead(c, len(self.spec)), c, self.mover)
        self.obj_head = TimedHead(ObjectHead(c, groups=NUSC_GROUPS, direction=cfg.object_direction, moving=cfg.object_moving),
                                  c, self.mover)
        if cfg.box_vel_from_field:                               # its own vel branch is not read: frozen, so no parameter goes without a
            self.obj_head.head.vel.requires_grad_(False)          # gradient (and the optimiser does not decay it)
        if cfg.refine:
            import random
            self.refiner = BoxRefiner(c, groups=NUSC_GROUPS, use_dt=True)
            self._rng = random.Random(0)                          # candidate jitter
        if cfg.aux_heads:                                           # the layouts BEVLoss expects for them
            self.aux_cam = SemanticHead(cfg.c_cam, len(layer_spec(False, True)))
            self.aux_rad = SemanticHead(cfg.c_rad, len(radar_aux_spec(cfg.radar_stationary)))
        self.c_trunk = c
        self._dist_idx = [i for i, s in enumerate(self.spec) if s[1] == "dist"]    # distance channels -> the soft line map
        self.spec_1m = layer_spec_1m(cfg.centreline_1m)
        self.head_1m = SemanticHead(c, len(self.spec_1m)) if self.spec_1m else None   # one pyramid level down, reads the sem head's moved state.
        # Built LAST: at one seed the init of every other module is the same with the option on or off (the random numbers it draws come after theirs)
        self.student = _build_student(cfg) if cfg.camera_student else None            # t22: the image encoder that stands in for the cached teachers of frame 0 (the cached model uses it)

    # ---- camera
    def camera_bevs(self, batch):
        """(B, F, c_cam, X, Y) camera BEV per frame."""
        img = batch["img"]
        B, Fn = img.shape[:2]
        if self.cfg.camera_off:
            return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=img.device)
        if self.cfg.camera_features != "dualbev":
            raise ValueError(f"camera_features={self.cfg.camera_features!r} reads cached features (CachedBEVFusionModel)")
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.freeze_camera):
            prob, ctx = self.camera.image_features(img.flatten(0, 1), batch["cam_vec"].flatten(0, 1))
        prob, ctx = prob.view(B, Fn, *prob.shape[1:]), ctx.view(B, Fn, *ctx.shape[1:])
        self._prob0 = prob[:, 0]                                  # depth of the current frame (lane branch)
        if self.cam_adapter is not None:
            ctx = self.cam_adapter(ctx.flatten(0, 1)).view_as(ctx)
        hw = img.shape[-2:]
        keys = self._lift_keys(batch, hw)
        bevs = []
        for b in range(B):                                       # one calibration per sample (scenes differ)
            vol = self.lift(ctx[b], prob[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], hw, batch["ida"], key=None if keys is None else keys[b])
            bevs.append(self.cam_reduce(vol))
        return torch.stack(bevs)

    @staticmethod
    def _lift_keys(batch, hw, ida_name="ida"):
        """the projection-table cache key of every sample's current-frame camera (lift_pd.lift_key), made from the CPU copies in batch["host"] (train.host_side; kept in the
        batch only when sync_free is on), so the lift does not wait for the GPU to read its calibration; None when the batch has no such copies (the lift makes the key itself).
        ida_name: the batch entry of the crop transform this lift uses ('ida', or 'ida2' for the BEVFusion source)"""
        host = batch.get("host")
        if not host or "K" not in host or "T_cam_car" not in host or ((ida_name in batch) != (ida_name in host)):
            return None
        Kh, Th, ida = host["K"], host["T_cam_car"], host.get(ida_name)
        return [lift_key(Kh[b, 0], Th[b, 0], hw, ida) for b in range(Kh.shape[0])]

    @staticmethod
    def _flip_any(batch):
        """True if a sample of the batch is mirrored: read from the CPU copy in batch["host"] when there is one (a `.any()` of the GPU tensor waits for the GPU)"""
        fl = batch.get("flip")
        if fl is None:
            return False
        hf = (batch.get("host") or {}).get("flip")
        return bool((fl if hf is None else hf).any())

    def camera_visibility(self, batch, like):
        """(B, 1, X, Y): the cells the current camera frame observes (LiftPD.visibility per sample) - the obs_mass of the
        camera branch's world memory, so the empty BEV outside the field of view does not overwrite what the memory saw
        (before 2026-10-26 the write used obs_mass = 1 everywhere)."""
        return Precompute.from_model(self).camera(batch).to(like)

    @staticmethod
    def _branch_temporal(block, bevs, steps, state, step, obs_mass=None):
        """bevs (B, F, C, X, Y) with frame 0 = current; steps[k]: list of (dt, v, omega) per-sample (B,) tensors from
        frame k to the current time (collate); one batched call, past frames oldest first; obs_mass (B, 1, X, Y) where
        the sensor observed (memory write mask) or None (everywhere)."""
        if block is None:
            return bevs[:, 0], state
        past = [(bevs[:, k], steps[k]) for k in range(bevs.shape[1] - 1, 0, -1)]
        out, new_state, _ = block(bevs[:, 0], past, state, step, obs_mass)
        return out, new_state

    @staticmethod
    def _streaming(frames):
        """stream mode = ONE frame per call: its past comes from state["buf"], not from the batch (the one place of this rule)"""
        return frames.shape[1] == 1

    def _buffer_in(self, frames, state, step):
        """the buffer this call continues from, None = cold start. Stream mode: state["buf"], the ego `step` since that frame is needed.
        Batch mode: 'pair' never (its window is the batch's own previous frame); 'recurrent' the previous call's buffer when there is a
        `step` (the previous keyframe is the frame before the oldest one: at stride 2 and n_cam 3 the chain has no gap, so training
        carries the fused state from keyframe to keyframe); without state or step the chain starts cold at the oldest frame."""
        buf = state.get("buf")
        if buf is None:
            return None
        if self._streaming(frames):
            if step is None:
                raise ValueError("stream mode with a buffered frame needs `step`: the ego step since the previous frame")
            return buf
        return buf if self.cfg.cam_buffer == "recurrent" and step is not None else None

    def _previous(self, frames, steps, buf, step):
        """(current, previous raw BEV) as (B, 2, C, X, Y) with the ego steps [[], previous -> current], and whether the previous frame is a real
        one: what the 'pair' window and the motion block read, in BOTH modes the frame just before the current one (batch: frame 1; stream:
        state["buf"]["raw"] with the caller's `step`), so that stream = batch. Cold start in the stream only: the current frame itself with a
        zero step (zero velocity evidence, mass 1) and no window."""
        if not self._streaming(frames):
            return frames[:, :2], steps[:2], True
        cur = frames[:, 0]
        if buf is None:
            z = cur.new_zeros(cur.shape[0])
            return torch.stack([cur, cur], 1), [[], [(z, z, z)]], False
        return torch.stack([cur, buf["raw"]], 1), [[], as_chain(step, cur)], True

    @staticmethod
    def _next_buffer(frames, fused, recurrent):
        """state["buf"] after this call: this frame's raw BEV (pair, recurrent) and its fused output (recurrent); not detached here (run_clip does)"""
        return {"raw": frames[:, 0], "fused": fused} if recurrent else {"raw": frames[:, 0]}

    @staticmethod
    def _step_to_oldest(gap, steps, frames):
        """the trainer's keyframe gap step (dt, v, omega) of the gap mean as the step to the OLDEST frame: dt less the time of the oldest
        frame's own chain (clamped at 0), v and omega kept"""
        dt, v, w = gap
        used = sum((s[0] for s in as_chain(steps[-1], frames)), frames.new_zeros(frames.shape[0]))
        return (dt - used).clamp(min=0), v, w

    def _unroll(self, frames, steps, cam_state, step, obs, buf=None):
        """cam_buffer 'recurrent': frames (B, F, C, X, Y), 0 = newest, through the temporal block oldest -> newest, every frame lifted once. A
        call fuses its frame with the previous call's fused output only (past = [(fused, ego chain between the two frames)]: moved by
        splat_window with the block's velocity head and mass gate); the world memory is read and written at every frame, as in a stream.
        The chain continues from buf (the previous call's fused output) or starts cold at the oldest frame. `step` is the ego step to the
        oldest frame: stream (one frame) as given (a triple or the sweep chain since the previous frame); batch the trainer's keyframe gap
        step (one triple) reduced by `_step_to_oldest`. Needs steps[k + 1] = the sweeps k + 1 -> k, then steps[k] (the frames of a sample are consecutive).
        -> (fused output of the newest frame, new cam state)"""
        block, B, Fn = self.cam_temporal, frames.shape[0], frames.shape[1]
        move = as_chain(step, frames)
        if Fn > 1 and move:
            if len(move) != 1:
                raise ValueError("a batch of several frames takes the keyframe gap step as ONE (dt, v, omega) triple, not a chain")
            move = [self._step_to_oldest(move[0], steps, frames)]
        fused = buf["fused"] if buf is not None else None
        for k in range(Fn - 1, -1, -1):
            if k < Fn - 1:                                   # frame k + 1 -> k: the first sweeps of frame k + 1's chain
                n = len(steps[k + 1]) - len(steps[k])
                if n < 0:
                    raise ValueError(f"frame {k + 1} has fewer ego steps ({len(steps[k + 1])}) than frame {k} ({len(steps[k])})")
                move = as_chain(steps[k + 1][:n], frames)
            fused, cam_state, _ = block(frames[:, k], [] if fused is None else [(fused, move)], cam_state, move, obs)
        return fused, cam_state

    def _camera_buffered(self, frames, steps, radar, state, step, obs):
        """cam_buffer 'pair' / 'recurrent' (t21). The motion block, when present, reads (current raw BEV, previous raw BEV) in both - never the
        fused state; its time baseline is the previous processed frame (0.167 s at stride 2), not the window's oldest frame, in batch mode
        too (frame 1), so stream = batch. 'pair': the window is ONE frame, the previous one (a batch of 2 frames is the window path with
        K = 1). 'recurrent': `_unroll`. radar None: no motion block (`advance`). -> (camera BEV, cam state, motion-block levels or None,
        the new state["buf"])"""
        Fn, mode = frames.shape[1], self.cfg.cam_buffer
        if mode == "pair" and Fn > 2:
            raise ValueError(f"cam_buffer 'pair' takes two frames (n_cam 2) or one (stream), the batch has {Fn}")
        buf = self._buffer_in(frames, state, step)
        pair, pair_steps, warm = self._previous(frames, steps, buf, step)
        v_levels = None
        if self.motion_block is not None and radar is not None:
            v_levels = self.motion_block(*block_inputs(pair, pair_steps, radar, self.grid))
        if mode == "pair":
            cam, cam_state = self._branch_temporal(self.cam_temporal, pair if warm else pair[:, :1], pair_steps, state.get("cam"), step, obs)
        else:
            cam, cam_state = self._unroll(frames, steps, state.get("cam"), step, obs, buf)
        return cam, cam_state, v_levels, self._next_buffer(frames, cam, mode == "recurrent")

    def advance(self, batch, state=None, step=None):
        """stream mode, a frame that needs no output (cam_buffer 'pair' / 'recurrent'): lift + the camera temporal block only - no motion block,
        radar, trunk or heads - and the new state, the one `forward` would return for the cam / buf entries; forward() on the next frame
        continues from it. One frame, `step` as in forward. Needs rad_temporal 'none' (its state would go stale); a mirror-augmented batch is
        refused (training only)."""
        if self.cfg.cam_buffer == "window" or self.rad_temporal is not None or self._flip_any(batch):
            raise ValueError("advance: cam_buffer 'pair' / 'recurrent', rad_temporal 'none' and no flipped sample")
        batch = Precompute.from_model(self)(dict(batch), ("vis_cam",))
        frames = self.camera_bevs(batch)
        self._student_side = None                                 # no loss follows an advance
        steps = [[tuple(torch.as_tensor(t, dtype=frames.dtype, device=frames.device) for t in s) for s in f] for f in batch["steps"]]
        _, cam_state, _, buf = self._camera_buffered(frames, steps, None, state or {}, step, batch["vis_cam"].to(frames))
        return dict(state or {}, cam=cam_state, buf=buf)

    def _refine_train(self, obj, feat, gts, dt):
        """S5c training step of one target time: candidates = top-k decoded boxes (detached) + jittered GT copies,
        matched to the GT (losses.refine_gt_boxes: unobserved or unsupervised boxes ignored) -> the refiner's steps and
        the targets in the layout of object_head.refine_loss (losses.BEVLoss adds it to the objects group)."""
        with torch.no_grad():
            dets = decode({k: v.detach().float().cpu() for k, v in obj.items()}, self.grid, NUSC_GROUPS,
                          k=self.cfg.refine_k)
        box, group, valid, gt, matched = refine_candidates(dets, [refine_gt_boxes(g, self.grid) for g in gts],
                                                           self.cfg.refine_k, self.cfg.refine_jitter, rng=self._rng,
                                                           groups=NUSC_GROUPS)
        dev = feat.device
        steps = self.refiner(feat, box.to(dev), group.to(dev), self.grid, dt)
        st = {k: torch.stack([s[k] for s in steps]) for k in ("box_in", "delta", "quality")}    # (n_iter, B, N, ...)
        return dict(st, gt=gt.to(dev), matched=matched.to(dev), valid=valid.to(dev), group=group.to(dev))

    @torch.no_grad()
    def boxes(self, res, dt=0.0, k=50, threshold=0.2, rescore=True):
        """Run time: decoded boxes at the target time dt of a forward result (list (B) of box dicts), refined by S5c
        when the model has a refiner (rescore: sqrt(score x predicted IoU))."""
        p = res["out"][float(dt)]                                # post-processing: select cells of the dense box output
        dets = boxes_from_maps({kk: v.detach() for kk, v in p["obj"].items()}, self.grid, NUSC_GROUPS, k=k, threshold=threshold)
        if not self.cfg.refine or not any(dets):
            return dets
        return refine_boxes(self.refiner, p["obj_feat"], dets, self.grid, p["obj_dt"], rescore)

    def forward(self, batch, dts=(0.0,), state=None, step=None, ego=None):
        """batch from nuscenes_dataset.collate (or the same layout); dts: the times to decode (s, either sign);
        state: per-branch temporal states of the previous call (sequence training), step: the ego step (dt, v, omega)
        since that call, scalars or per-sample (B,) tensors. cam_buffer 'pair' / 'recurrent' (t21): the state also holds "buf" (the previous
        frame's raw camera BEV, and its fused output); a batch of ONE frame is a stream step: the previous frame comes from state["buf"] (none:
        cold start) and `step` is the ego step since it, one triple or the per-sweep chain (a list of triples like batch["steps"][k], which
        is what the batch path uses); with 'recurrent' a batch of several frames continues from state["buf"] too when `step` is given (the
        trainer's keyframe gap step), 'pair' never. Returns {"out": preds, "state": ..., "features": trunk
        features}, preds in the layout of losses.BEVLoss: {dt: {"sem", "obj" (object head maps), "mass" (splat mass at dt), "v_hat" (dt = 0),
        "cam_aux" / "radar_aux" (dt = 0, training only)}}. ego: optional {nominal dt: (dt, v, omega)} as (B,) tensors -
        the true gap to the target keyframe and the gap-mean ego motion per sample (the GT keyframe is 0.40-0.60 s
        away, not the nominal 0.5 s); without it a head decodes at the nominal dt with the batch's v, omega. The output
        key stays the nominal dt."""
        with LS.enabled(splat=self.cfg.splat_fused, sync=self.cfg.sync_free):       # D53: the launch-saving variants, for this call only
            return self._forward(batch, dts, state, step, ego)

    def _forward(self, batch, dts, state, step, ego):
        state = state or {}
        if "host" in batch and not self.cfg.sync_free:           # D54: the CPU copies of the calibration and the flip flags are used only with sync_free; without it the old code runs
            batch = {k: v for k, v in batch.items() if k != "host"}
        need = (("vis_cam",) if self.cam_temporal is not None else ()) + \
               ((self._rad_mask,) if self.rad_temporal is not None else ())     # every memory kind gets its write mask (arms matched)
        batch = Precompute.from_model(self)(dict(batch), need) if need else batch   # layer 1: only the maps a memory reads, in
        cam_frames = self.camera_bevs(batch)                                         # the batch's frame (kept when precomputed)
        student_side, self._student_side = self._student_side, None                  # camera_student: set by the cached model's camera_bevs, handed out with the result
        steps = [[tuple(torch.as_tensor(t, dtype=cam_frames.dtype, device=cam_frames.device) for t in s) for s in f]
                 for f in batch["steps"]]
        cam_obs = batch["vis_cam"].to(cam_frames) if "vis_cam" in need else None
        flip = batch.get("flip")                                 # mirror augmentation (training/augment.py): the other inputs
        flipped = self._flip_any(batch)
        if flipped:                                              # and the targets of these samples were mirrored by flip_batch
            cam_frames = flip_cam(cam_frames, None, flip)[0]     # (the visibility map is mirrored by the precompute step)
        v_levels, buf = None, None
        if self.cfg.cam_buffer != "window":                      # t21: one buffered frame instead of the window (_camera_buffered)
            cam, cam_state, v_levels, buf = self._camera_buffered(cam_frames, steps, batch["radar"], state, step, cam_obs)
        else:
            if self.motion_block is not None:                    # t15: the velocity field from two camera frames + radar (motion_block.py)
                v_levels = self.motion_block(*block_inputs(cam_frames, steps, batch["radar"], self.grid))
            cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step, cam_obs)
        if self.lane is not None:                                # t7: current frame at stride 4, zero start, mirrored like cam; the depth, K, ida
                                                                 # are those of source 1 (DualBEV; with 'dual' also what batch["prob"], ["ida"] hold)
            img0 = batch["lane_img"] if "lane_img" in batch else batch["img"][:, 0]
            prob0 = batch["prob"][:, 0].float() if "prob" in batch else self._prob0
            hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
            lane = self.lane(img0, prob0, batch["K"][:, 0], batch["T_cam_car"][:, 0], hw, batch["ida"], keys=self._lift_keys(batch, hw))
            if flipped:
                lane = flip_cam(lane[:, None], None, flip)[0][:, 0]
            cam = cam + lane
        radar_side = None
        if self.cfg.radar_kind == "off":
            rad = torch.zeros(cam.shape[0], self.cfg.c_rad, self.grid.nx, self.grid.ny, device=cam.device)
        elif self.cfg.radar_kind == "delta_slots":               # features + the lossless raster: the side loss (train.py) reads both
            rad, rstate = self.radar(batch["radar"], cam.shape[0], with_state=True)
            radar_side = {"features": rad, "state": rstate}
        else:
            rad = self.radar(batch["radar"], cam.shape[0])
        rad_obs = batch[self._rad_mask].to(rad) if self._rad_mask in batch else None   # radar memory: write where it measured (with
                                                                                       # rad_static_gate: where it measured the static world);
                                                                                       # no radar points: no mask = everywhere
        rad, rad_state = self._branch_temporal(self.rad_temporal, rad[:, None], [[]], state.get("rad"), step, rad_obs)
        S = self.trunk.features(cam, rad) if self.fusion is None else self.trunk.features(self.fusion(cam, rad))
        if self.head_reduce is not None:
            S = self.head_reduce(S)
        v, w = batch["v"].to(S), batch["omega"].to(S)
        out = {}
        refine_gt = (sample_targets(batch["gt"]) if self.cfg.refine and self.training and batch.get("gt") else {})
        for dt in dts:
            e = (ego or {}).get(float(dt))
            dt_i, v_i, w_i = (dt, v, w) if e is None else (torch.as_tensor(x).to(S) for x in e)
            vf = v_levels[-1] if v_levels is not None else None
            carry = vf if (vf is not None and self.cfg.box_vel_from_field and abs(float(dt)) > 1e-6) else None
            if self.cfg.share_mover:                             # D53: one splat per dt for both heads (with a carry the object head's splat carries the state too)
                pre = self.mover(S, dt_i, v_i, w_i, v_field=vf, carry=carry)
                pre = pre if carry is not None else pre + (None,)
                sem, a = self.sem_head(S, dt_i, v_i, w_i, return_aux=True, premoved=pre[:3] + (None,))
                obj, ao = self.obj_head(S, dt_i, v_i, w_i, return_aux=True, premoved=pre)
            else:
                sem, a = self.sem_head(S, dt_i, v_i, w_i, return_aux=True, v_field=vf)
                obj, ao = self.obj_head(S, dt_i, v_i, w_i, return_aux=True, v_field=vf, carry=carry)
            if vf is not None and self.cfg.box_vel_from_field:   # the box velocity is the field (t12's second decision)
                obj = dict(obj, vel=vf if carry is None else _turn(ao["carry"], w_i, dt_i))
            p = {"sem": sem, "obj": obj, "mass": a["mass"]}
            if self.head_1m is not None:                         # the 1 m level: the state the sem head decoded at t + dt, average-pooled
                p["sem_1m"] = self.head_1m(F.avg_pool2d(a["state"], LEVELS["1m"]))
            if self.cfg.refine:
                p["obj_feat"] = ao["state"]                          # what the refiner samples (also at run time)
                p["obj_dt"] = torch.as_tensor(dt_i).to(S).reshape(-1).expand(S.shape[0])
                if len(refine_gt.get(float(dt), [])) == S.shape[0]:
                    p["refine"] = self._refine_train(obj, ao["state"], refine_gt[float(dt)], dt_i)
            if self.cfg.dense_outputs:                           # output interface: dense, differentiable boxes and lines (extract.py)
                p["boxes"] = dense_boxes({k: v.float() for k, v in obj.items()}, self.grid, NUSC_GROUPS)
                p["lines"] = soft_line_map(sem[:, self._dist_idx].float().clamp(min=0), self.grid.res)   # clamped as the evaluation does
            if abs(float(dt)) < 1e-6:
                p["v_hat"] = self.mover.motion(S) if v_levels is None else v_levels[-1]
                if v_levels is not None and len(v_levels) > 1:     # the coarser levels, coarse to fine: the same motion term (deep supervision,
                    p["v_levels"] = {f"l{i}": v for i, v in enumerate(v_levels[:-1])}   # losses.py); a dict: the trainer casts tensors / dicts
                if self.cfg.aux_heads and self.training:
                    p["cam_aux"], p["radar_aux"] = self.aux_cam(cam), self.aux_rad(rad)
            out[float(dt)] = p
        res = {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}
        if buf is not None:
            res["state"]["buf"] = buf
        if radar_side is not None:
            res["radar_side"] = radar_side
        if student_side is not None:                              # t22: the anchor of train.forward_loss reads it (student_encoder.anchor_terms)
            res["student_side"] = student_side
        return res


def _turn(vec, omega, dt):
    """velocity vectors (B, 2, X, Y) carried to t + dt: turned into the ego axes of the target time, R(-omega dt) (the rotation of
    splat.splat_positions); omega, dt scalars or (B,)"""
    a = -(torch.as_tensor(omega).to(vec) * torch.as_tensor(dt).to(vec)).reshape(-1, 1, 1)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1]], 1)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from losses import BEVLoss, sample_targets
    from nuscenes_dataset import FrontRadarDataset, collate
    torch.manual_seed(0)
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))        # the bev_fusion folder
    assert BEVFusionModel(ModelConfig(pretrained="")).fusion.p_drop == 0.2
    for off in (dict(camera_off=True), dict(radar_kind="off")):    # one sensor off: no modality dropout (it would blank the sample)
        assert BEVFusionModel(ModelConfig(pretrained="", **off)).fusion.p_drop == 0.0, off
    try:
        BEVFusionModel(ModelConfig(pretrained="", camera_off=True, lane_branch="rgb"))
        raise AssertionError("lane_branch with camera_off was accepted")
    except ValueError:
        pass
    ds = FrontRadarDataset(os.path.expanduser("~/data/nuscenes"), "v1.0-mini", n_cam=3, cam_stride=2, with_gt=True)
    batch = collate([ds[10], ds[200]])                       # mid-scene keyframes: GT of k-1, k, k+1 exist
    gts = sample_targets(batch["gt"])
    dts = tuple(sorted(gts))
    assert dts == (-0.5, 0.0, 0.5) and all(len(g) == 2 for g in gts.values()), {k: len(g) for k, g in gts.items()}
    model = BEVFusionModel()
    model.train()
    res = model(batch, dts=dts)
    o = res["out"][0.0]
    X, Y = model.grid.nx, model.grid.ny
    assert o["sem"].shape == (2, len(model.spec), X, Y) and o["obj"]["hm"].shape == (2, 3, X, Y), o["sem"].shape
    assert set(res["out"]) == set(dts) and "v_hat" in o and "cam_aux" in o and "v_hat" not in res["out"][0.5]
    crit = BEVLoss(grid=model.grid)
    L = crit(res["out"], gts, step_frac=1.0)
    assert torch.isfinite(L["total"]), {k: float(v) for k, v in L.items() if torch.is_tensor(v)}
    L["total"].backward()
    assert model.camera.depth_net.depth_conv[4].weight.grad is None, "frozen camera heads get no gradient"
    for name, mod in (("reduce", model.cam_reduce), ("camera temporal", model.cam_temporal), ("radar", model.radar),
                      ("fusion", model.fusion), ("trunk", model.trunk), ("time mover", model.mover),
                      ("semantic head", model.sem_head), ("object head", model.obj_head)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), f"no gradient: {name}"
    n_train = sum(p.numel() for p in trainable_parameters(model))
    n_all = sum(p.numel() for p in model.parameters())
    groups = ", ".join(f"{k[6:]} {float(v.detach()):.2f}" for k, v in L.items() if k.startswith("group."))
    # a second call with the returned state and the step since (sequence mode)
    with torch.no_grad():
        model.eval()
        res2 = model(batch, state=res["state"], step=(torch.full((2,), 1 / 12), torch.tensor([5.0, 8.0]),
                                                      torch.tensor([0.0, 0.1])))
    assert res2["out"][0.0]["sem"].shape == o["sem"].shape and "cam_aux" not in res2["out"][0.0]
    # the camera memory writes only where the camera observes (obs_mass = LiftPD.visibility), nothing outside the view
    if isinstance(model.cam_temporal, TemporalFusion):
        with torch.no_grad():
            vis = model.camera_visibility(batch, o["sem"])                                  # (B, 1, X, Y)
            st = model(batch)["state"]["cam"]
            from temporal_fusion import read_memory as _rm
            _, conf = _rm(st["M"], st["W"], st["pose"], model.cam_temporal.ego_pts, vis.shape[-2], vis.shape[-1],
                          model.cam_temporal.mem_grid)
        assert 0.05 < float(vis.mean()) < 0.95, f"visibility covers part of the grid ({float(vis.mean()):.2f})"
        assert float(conf[vis < 0.5].max()) < 1e-4 and float(conf[vis > 0.5].mean()) > 0.1, \
            "memory written inside the camera view only"
    # per-sample true keyframe gap: the ego path equals the scalar path for shared values; a 0.45 s gap moves 0.9 x
    with torch.no_grad():
        vv, ww = batch["v"].float(), batch["omega"].float()
        base = model(batch, dts=(0.5,))["out"][0.5]
        same = model(batch, dts=(0.5,), ego={0.5: (torch.full((2,), 0.5), vv, ww)})["out"][0.5]
        assert torch.allclose(same["sem"], base["sem"], atol=1e-5) and torch.allclose(same["mass"], base["mass"])
        v10, w0 = torch.full((2,), 10.0), torch.zeros(2)
        unk = lambda gaps: (model(batch, dts=(0.5,), ego={0.5: (torch.tensor(gaps), v10, w0)})["out"][0.5]["mass"]
                            < 0.5).float().mean((1, 2, 3))
        u_ref, u_gap = unk([0.5, 0.5]), unk([0.5, 0.45])
    assert abs(float(u_gap[0] - u_ref[0])) < 1e-6 and abs(float(u_gap[1] / u_ref[1]) - 0.9) < 0.05, (u_ref, u_gap)
    del model, res, res2
    # S5c in training: refine=True adds the refinement terms to the objects group, the gradient reaches the refiner,
    # the output survives the trainer's float cast; at run time boxes() returns refined boxes
    m = BEVFusionModel(ModelConfig(pretrained="", refine=True)).train()
    r = m(batch, dts=dts)
    cast = {dt: {k: ({a: b.float() for a, b in v.items()} if isinstance(v, dict) else v.float()) for k, v in p.items()}
            for dt, p in r["out"].items()}                                       # as train.py does
    Lr = BEVLoss(grid=m.grid)(cast, gts, step_frac=1.0)
    assert all(f"refine.reg@{dt:+.1f}" in Lr for dt in dts) and torch.isfinite(Lr["total"]), sorted(Lr)[:8]
    Lr["total"].backward()
    assert any(q.grad is not None and q.grad.abs().sum() > 0 for q in m.refiner.mlp.parameters()), "refiner gradient"
    n_ref = int(r["out"][0.0]["refine"]["valid"].sum())
    m.eval()
    with torch.no_grad():
        rb = m.boxes(m(batch, dts=(0.0,)), 0.0, threshold=0.0)
    assert len(rb) == 2 and all(len(b) > 0 for b in rb) and "obj_feat" in r["out"][0.0]
    del m, r, cast, Lr
    # every swappable block (shapes only: no checkpoint, no gradient)
    swaps = [{"lift": "p"}, {"fusion": "gate"}, {"cam_temporal": "none"}, {"rad_temporal": "window_memory"},
             {"lift": "s"}, {"cam_temporal": "ego_gru"}, {"rad_temporal": "ego_gru"}, {"fusion": "cmx"},
             {"radar_kind": "pillars"}, {"radar_kind": "delta_slots", "c_rad": 64}, {"radar_kind": "off"}, {"trunk_kind": "second", "head_channels": 96},
             {"camera_off": True}, {"cam_adapter": True}, {"cam_collapse": "zsum"}, {"cam_buffer": "recurrent"}]
    tried = []
    for kw in swaps:
        m = BEVFusionModel(ModelConfig(pretrained="", **kw)).eval()
        with torch.no_grad():
            r = m(batch, dts=(0.0, 0.5))
        assert r["out"][0.5]["sem"].shape == o["sem"].shape, kw
        tried.append("=".join(str(x) for x in next(iter(kw.items()))))
        del m, r
    print(f"model selftest ok: two mini samples through the whole chain at dt {dts}, BEVLoss total "
          f"{float(L['total'].detach()):.2f} ({groups}); {n_train / 1e6:.1f} M of {n_all / 1e6:.1f} M parameters trainable "
          f"(camera heads frozen, no gradient); gradients reach reduce, camera temporal, radar, fusion, trunk, mover, "
          f"both heads; sequence mode with state; per-sample keyframe gap (ego = scalar path; 0.45 s gap: unknown band "
          f"{float(u_gap[1] / u_ref[1]):.2f} x the 0.5 s band); S5c refinement trains ({n_ref} valid candidates at dt 0, "
          f"gradient reaches the refiner, float cast ok) and refines at run time; swapped blocks run: {', '.join(tried)}")


if __name__ == "__main__":
    _selftest()
