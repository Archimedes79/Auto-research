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
(boxes(): extract.select_boxes = object_head.decode).
Every block is chosen in ModelConfig; each option is either the plan's design or its paper fallback:
  lift          "pd"  lift PD, pull with depth weight (DualBEV / FastBEV++ / FB-BEV) - default
                "p"   pull without depth (Simple-BEV, Fast-BEV) - control
                "s"   forward splat (LSS / BEVPoolV2) - training/fallbacks.py LiftS
  cam_temporal  "window_memory"  training/temporal_fusion.py TemporalFusion (window of original frames + world memory)
                "ego_gru"        FIERY-style ego warp + ConvGRU - training/fallbacks.py EgoWarpGRU (control arm A)
                "none"           current frame only
  rad_temporal  "none" (default: the 0.5 s of sweeps are already accumulated at point level), "window_memory", "ego_gru"
  fusion        "concat" (BEVFusion ConvFuser) - default, "gate" (GaussianCaR FeatureRectify), "cmx" (GaussianCaR's
                4-stage fuser + decoder, training/fallbacks.py CMXTrunk: replaces fusion AND backbone)
  heads         semantic (channels = losses.layer_spec) and objects (CenterPoint, S5a), both decoded at any dt by
                training/timed_head.py (the shared TimeMover moves S to dt); the output is the prediction layout of losses.BEVLoss;
                S5c refinement works on the decoded boxes (object_head.py). forward(ego=...) takes the true gap per sample.
Conventions as the blocks: grid x 0..60 m forward, y -26..26 m left, 0.5 m; maps (B, C, X, Y).
Run the self-test (nuScenes mini, 2 samples): python training/model.py
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

try:
    from .augment import flip_cam
    from .bev_trunk import BEVTrunk, SemanticHead
    from .blocks_new import CtxAdapter, RadarPillars, SecondTrunk, ZSumReduce
    from .camera_encoder import DEPTH_BINS, FrontCameraEncoder, depth_centres
    from .extract import dense_boxes, select_boxes, soft_line_map
    from .fusion import ConcatFusion, GateFusion
    from .lane_branch import LaneBranch
    from .lift_pd import Grid, LiftPD
    from .losses import NUSC_GROUPS, layer_spec, radar_aux_spec, refine_gt_boxes, sample_targets
    from .object_head import BoxRefiner, ObjectHead, decode, refine_boxes, refine_candidates
    from .precompute import Precompute
    from .radar_bev import RadarBEV
    from .radar_encoder import RadarEncoder, n_channels
    from .temporal_fusion import TemporalFusion
    from .timed_head import TimedHead, TimeMover
except ImportError:                                   # run as a script
    from augment import flip_cam
    from bev_trunk import BEVTrunk, SemanticHead
    from blocks_new import CtxAdapter, RadarPillars, SecondTrunk, ZSumReduce
    from camera_encoder import DEPTH_BINS, FrontCameraEncoder, depth_centres
    from extract import dense_boxes, select_boxes, soft_line_map
    from fusion import ConcatFusion, GateFusion
    from lane_branch import LaneBranch
    from lift_pd import Grid, LiftPD
    from losses import NUSC_GROUPS, layer_spec, radar_aux_spec, refine_gt_boxes, sample_targets
    from object_head import BoxRefiner, ObjectHead, decode, refine_boxes, refine_candidates
    from precompute import Precompute
    from radar_bev import RadarBEV
    from radar_encoder import RadarEncoder, n_channels
    from temporal_fusion import TemporalFusion
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
    rad_temporal: str = "none"
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
                                       # block (lane_branch.py): "" = off | "res4" (frozen ResNet-50 layer 1) | "rgb" (CNN from scratch)
    lane_pretrained: str = "models/pretrained/dualbev-r50-cbgs.pth"    # weights of the "res4" features
    oversample: int = 4
    centrelines: bool = False          # head layout = losses.layer_spec(centrelines, dense_objects)
    dense_objects: bool = True
    radar_stationary: bool = True      # radar aux head = losses.radar_aux_spec(radar_stationary)
    n_sem: int = None                  # optional check: must equal len(layer_spec(...)) when given (train.py passes it)
    aux_heads: bool = True
    object_direction: bool = False     # object head: sign-free axis + front / back class (object_head.ObjectHead)
    object_moving: bool = False        # object head: moving / static logit, velocity on moving centres only (t11 hurdle)
    motion_block: bool = False         # t15: the time mover's velocity field from training/motion_block.py (the current and the oldest
                                       # camera frame + the radar Doppler, a 3-level pyramid) instead of the 1x1 head on the trunk; needs
                                       # >= 2 camera frames (n_cam) and a radar
    motion_strides: tuple = (4, 2, 1)  # its pyramid levels in cells, coarse to fine; (4, 2) = the runtime arm without the 0.5 m level
    box_vel_from_field: bool = False   # t15: the object head's `vel` = the motion block's field (at dt != 0 moved with the state, the
                                       # vector turned with the ego yaw); the object head's own vel branch stays (unused for the output)
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
FUSIONS = {"concat": ConcatFusion, "gate": GateFusion, "cmx": None}  # cmx: the backbone takes both branch BEVs
TRUNKS = {"dualbev": _dualbev_trunk, "second": _second_trunk}
BACKBONES = dict(TRUNKS, cmx=_cmx_trunk)


def check_config(cfg):
    """every unknown kind and every combination of slot kinds that does not fit, in one place (one ValueError naming all)"""
    errs = [f"{name}={val!r}: one of {sorted(table)}" for name, val, table in
            (("camera_features", cfg.camera_features, CAMERAS), ("lift", cfg.lift, LIFTS), ("cam_collapse", cfg.cam_collapse, COLLAPSES),
             ("cam_temporal", cfg.cam_temporal, TEMPORALS), ("rad_temporal", cfg.rad_temporal, TEMPORALS),
             ("radar_kind", cfg.radar_kind, RADARS), ("fusion", cfg.fusion, FUSIONS), ("trunk_kind", cfg.trunk_kind, TRUNKS))
            if val not in table]
    if len(CAMERAS.get(cfg.camera_features, ())) > 1 and (cfg.lift == "s" or cfg.cam_adapter):
        errs.append(f"camera_features={cfg.camera_features!r} (several feature sources) is not supported with lift 's' or cam_adapter")
    if cfg.camera_features == "bevfusion" and cfg.lift == "s":
        errs.append("camera_features='bevfusion' is not supported with lift 's'")
    if cfg.lane_branch and (cfg.camera_features != "dualbev" or cfg.lift == "s"):
        errs.append("lane_branch needs camera_features 'dualbev' and lift 'pd' / 'p'")
    if cfg.radar_kind == "delta_slots" and cfg.c_rad < n_channels():
        errs.append(f"radar_kind='delta_slots' needs c_rad >= {n_channels()} (one channel per input channel: the pass-through start); c_rad = {cfg.c_rad}")
    if cfg.lane_branch and cfg.camera_off:                            # the branch reads the current image: a "camera off" arm would still see it
        errs.append("lane_branch reads the camera image: it cannot be combined with camera_off")
    if cfg.motion_block and (cfg.camera_off or cfg.radar_kind == "off"):
        errs.append("motion_block reads two camera frames and the radar Doppler: it needs the camera and a radar")
    if cfg.box_vel_from_field and not cfg.motion_block:
        errs.append("box_vel_from_field reads the motion block's field: it needs motion_block")
    if cfg.motion_block and (not cfg.motion_strides or list(cfg.motion_strides) != sorted(cfg.motion_strides, reverse=True)):
        errs.append(f"motion_strides must be a non-empty decreasing tuple of cell strides, is {cfg.motion_strides}")
    if errs:
        raise ValueError("; ".join(errs))


class BEVFusionModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        check_config(cfg)
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
        if cfg.motion_block:                                     # t15: the field comes from the block; the 1x1 head is not built (it would
            from motion_block import MotionBlock                 # get no gradient); TimeMover then needs v_field on every call
            self.motion_block = MotionBlock(cfg.c_cam, self.grid, strides=tuple(cfg.motion_strides))
            self.mover.motion = None
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
        bevs = []
        for b in range(B):                                       # one calibration per sample (scenes differ)
            vol = self.lift(ctx[b], prob[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], hw, batch["ida"])
            bevs.append(self.cam_reduce(vol))
        return torch.stack(bevs)

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
        dense = dense_boxes({kk: v.double().cpu() for kk, v in p["obj"].items()}, self.grid, NUSC_GROUPS, soft=False)   # double: = decode
        dets = select_boxes(dense, NUSC_GROUPS, k=k, threshold=threshold)
        if not self.cfg.refine or not any(dets):
            return dets
        return refine_boxes(self.refiner, p["obj_feat"], dets, self.grid, p["obj_dt"], rescore)

    def forward(self, batch, dts=(0.0,), state=None, step=None, ego=None):
        """batch from nuscenes_dataset.collate (or the same layout); dts: the times to decode (s, either sign);
        state: per-branch temporal states of the previous call (sequence training), step: the ego step (dt, v, omega)
        since that call, scalars or per-sample (B,) tensors. Returns {"out": preds, "state": ..., "features": trunk
        features}, preds in the layout of losses.BEVLoss: {dt: {"sem", "obj" (object head maps), "mass" (splat mass at dt), "v_hat" (dt = 0),
        "cam_aux" / "radar_aux" (dt = 0, training only)}}. ego: optional {nominal dt: (dt, v, omega)} as (B,) tensors -
        the true gap to the target keyframe and the gap-mean ego motion per sample (the GT keyframe is 0.40-0.60 s
        away, not the nominal 0.5 s); without it a head decodes at the nominal dt with the batch's v, omega. The output
        key stays the nominal dt."""
        state = state or {}
        need = (("vis_cam",) if self.cam_temporal is not None else ()) + \
               (("radar_hits",) if self.rad_temporal is not None else ())     # every memory kind gets its write mask (arms matched)
        batch = Precompute.from_model(self)(dict(batch), need) if need else batch   # layer 1: only the maps a memory reads, in
        cam_frames = self.camera_bevs(batch)                                         # the batch's frame (kept when precomputed)
        steps = [[tuple(torch.as_tensor(t, dtype=cam_frames.dtype, device=cam_frames.device) for t in s) for s in f]
                 for f in batch["steps"]]
        cam_obs = batch["vis_cam"].to(cam_frames) if "vis_cam" in need else None
        flip = batch.get("flip")                                 # mirror augmentation (training/augment.py): the other inputs
        if flip is not None and bool(flip.any()):                # and the targets of these samples were mirrored by flip_batch
            cam_frames = flip_cam(cam_frames, None, flip)[0]     # (the visibility map is mirrored by the precompute step)
        v_levels = None
        if self.motion_block is not None:                        # t15: the velocity field from two camera frames + radar (motion_block.py)
            from motion_block import block_inputs
            v_levels = self.motion_block(*block_inputs(cam_frames, steps, batch["radar"], self.grid))
        cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step, cam_obs)
        if self.lane is not None:                                # t7: current frame at stride 4, zero start, mirrored like cam
            img0 = batch["lane_img"] if "lane_img" in batch else batch["img"][:, 0]
            prob0 = batch["prob"][:, 0].float() if "prob" in batch else self._prob0
            hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
            lane = self.lane(img0, prob0, batch["K"][:, 0], batch["T_cam_car"][:, 0], hw, batch["ida"])
            if flip is not None and bool(flip.any()):
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
        rad_obs = batch["radar_hits"].to(rad) if "radar_hits" in batch else None   # radar memory: write where it measured (no
                                                                                   # radar points: no mask = everywhere)
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
            sem, a = self.sem_head(S, dt_i, v_i, w_i, return_aux=True, v_field=vf)
            carry = vf if (vf is not None and self.cfg.box_vel_from_field and abs(float(dt)) > 1e-6) else None
            obj, ao = self.obj_head(S, dt_i, v_i, w_i, return_aux=True, v_field=vf, carry=carry)
            if vf is not None and self.cfg.box_vel_from_field:   # the box velocity is the field (t12's second decision)
                obj = dict(obj, vel=vf if carry is None else _turn(ao["carry"], w_i, dt_i))
            p = {"sem": sem, "obj": obj, "mass": a["mass"]}
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
                if v_levels is not None and len(v_levels) > 1:
                    p["v_levels"] = v_levels[:-1]                  # the coarser levels: the same motion term (deep supervision, losses.py)
                if self.cfg.aux_heads and self.training:
                    p["cam_aux"], p["radar_aux"] = self.aux_cam(cam), self.aux_rad(rad)
            out[float(dt)] = p
        res = {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}
        if radar_side is not None:
            res["radar_side"] = radar_side
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
             {"camera_off": True}, {"cam_adapter": True}, {"cam_collapse": "zsum"}]
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
