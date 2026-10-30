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

Every block is chosen in ModelConfig; each option is either the plan's design or its paper fallback:
  lift          "pd"  lift PD, pull with depth weight (DualBEV / FastBEV++ / FB-BEV) - default
                "p"   pull without depth (Simple-BEV, Fast-BEV) - control
                "s"   forward splat (LSS / BEVPoolV2) - training/fallbacks.py LiftS
  cam_temporal  "window_memory"  training/temporal_fusion.py TemporalFusion (window of original frames + world memory)
                "ego_gru"        FIERY-style ego warp + ConvGRU - training/fallbacks.py EgoWarpGRU (control arm A)
                "none"           current frame only
  rad_temporal  "none" (default: the 0.5 s of sweeps are already accumulated at point level), "window_memory", "ego_gru"
  fusion        "concat" (BEVFusion ConvFuser) - default, "gate" (GaussianCaR FeatureRectify), "cmx" (GaussianCaR's
                4-stage fuser, training/fallbacks.py CMXFuser)
  heads         semantic (channels = losses.layer_spec) and objects (CenterPoint, S5a), both decoded at any dt by
                training/timed_head.py; the output is the prediction layout of losses.BEVLoss; S5b box propagation and
                S5c refinement work on the decoded boxes (object_head.py)
Open: the heads take one dt per call, so a batch is decoded at the nominal dt (0, +-0.5 s); the actual keyframe gap
(0.45-0.55 s) and a missing k-1 / k+1 at scene ends need a per-sample dt and a GT mask before random batches.
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
    from .fusion import ConcatFusion, GateFusion
    from .lift_pd import Grid, LiftPD, camera_visibility
    from .losses import NUSC_GROUPS, layer_spec, radar_aux_spec, refine_gt_boxes, sample_targets
    from .object_head import BoxRefiner, ObjectHead, decode, refine_boxes, refine_candidates
    from .radar_bev import RadarBEV
    from .temporal_fusion import TemporalFusion
    from .timed_head import TimedHead, TimeMover
except ImportError:                                   # run as a script
    from augment import flip_cam
    from bev_trunk import BEVTrunk, SemanticHead
    from blocks_new import CtxAdapter, RadarPillars, SecondTrunk, ZSumReduce
    from camera_encoder import DEPTH_BINS, FrontCameraEncoder, depth_centres
    from fusion import ConcatFusion, GateFusion
    from lift_pd import Grid, LiftPD, camera_visibility
    from losses import NUSC_GROUPS, layer_spec, radar_aux_spec, refine_gt_boxes, sample_targets
    from object_head import BoxRefiner, ObjectHead, decode, refine_boxes, refine_candidates
    from radar_bev import RadarBEV
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
    cam_collapse: str = "zconv"        # try 7: "zconv" (1x1 conv over Z x 80 channels) | "zsum" (sum over heights, then 1x1; blocks_new.ZSumReduce)
    cam_adapter: bool = False          # try 6 (t3_improve_loop): trainable residual adapter on the image context features before the lift
    camera_off: bool = False           # study arm: the camera BEV is zero (radar-only system), the image heads are not run
    oversample: int = 4
    centrelines: bool = False          # head layout = losses.layer_spec(centrelines, dense_objects)
    dense_objects: bool = True
    radar_stationary: bool = True      # radar aux head = losses.radar_aux_spec(radar_stationary)
    n_sem: int = None                  # optional check: must equal len(layer_spec(...)) when given (train.py passes it)
    aux_heads: bool = True
    object_direction: bool = False     # object head: sign-free axis + front / back class (object_head.ObjectHead)
    refine: bool = False               # S5c: deformable box refinement (object_head.BoxRefiner, n = 1, dt-aware) on the
                                       # object head's features at the target time; trained on decoded + jittered GT
    refine_k: int = 50                 # candidates: top-k decoded boxes per sample + refine_jitter copies of each GT box
    refine_jitter: int = 1
    pretrained: str = "models/pretrained/dualbev-r50-cbgs.pth"
    freeze_camera: bool = True
    z_levels: tuple = Z_LEVELS
    grid_kwargs: dict = field(default_factory=dict)


def _temporal(kind, c, grid):
    if kind == "window_memory":
        return TemporalFusion(c, c, grid)
    if kind == "ego_gru":
        return _fallbacks().EgoWarpGRU(c, c, grid)
    if kind == "none":
        return None
    raise ValueError(kind)


class BEVFusionModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.grid = Grid(z_levels=cfg.z_levels, **cfg.grid_kwargs)
        # camera branch: pre-trained image heads (only image_features is used) + our lift + reduce
        self.camera = FrontCameraEncoder()
        if cfg.pretrained:
            rep = self.camera.load_pretrained(cfg.pretrained)
            if rep["missing"] or rep["shape_mismatch"]:
                raise RuntimeError(f"camera weights do not fit: {rep}")
        if cfg.freeze_camera:
            self.camera.freeze_camera_heads()
        del self.camera.lift, self.camera.out_conv                   # replaced by the model's own lift
        n_ctx, nz = 80, len(cfg.z_levels)
        if cfg.lift in ("pd", "p"):
            self.lift = LiftPD(self.grid, depth_centres(), use_depth=(cfg.lift == "pd"), oversample=cfg.oversample)
        elif cfg.lift == "s":
            self.lift = _fallbacks().LiftS(self.grid, depth_centres())
        else:
            raise ValueError(cfg.lift)
        self.cam_adapter = CtxAdapter(n_ctx) if cfg.cam_adapter else None
        if cfg.cam_collapse == "zsum":
            self.cam_reduce = ZSumReduce(nz, n_ctx, cfg.c_cam)
        elif cfg.cam_collapse == "zconv":
            self.cam_reduce = nn.Sequential(nn.Conv2d(nz * n_ctx, cfg.c_cam, 1, bias=False), nn.BatchNorm2d(cfg.c_cam),
                                            nn.ReLU(True))
        else:
            raise ValueError(cfg.cam_collapse)
        self.cam_temporal = _temporal(cfg.cam_temporal, cfg.c_cam, self.grid)
        # radar branch
        if cfg.radar_kind == "pillars":
            self.radar = RadarPillars(self.grid, cfg.c_rad)
        elif cfg.radar_kind in ("raster", "off"):
            self.radar = RadarBEV(self.grid, cfg.c_rad)
        else:
            raise ValueError(cfg.radar_kind)
        self.rad_temporal = _temporal(cfg.rad_temporal, cfg.c_rad, self.grid)
        # fusion, trunk (CMX is its own four-stage pyramid + decoder: it replaces both)
        if cfg.fusion == "cmx":
            self.fusion = None
            c = 128
            self.trunk = _fallbacks().CMXTrunk(cfg.c_cam, cfg.c_rad, embed=cfg.c_fuse, c_out=c)
        else:
            if cfg.fusion == "concat":
                self.fusion = ConcatFusion(cfg.c_cam, cfg.c_rad, cfg.c_fuse)
            elif cfg.fusion == "gate":
                self.fusion = GateFusion(cfg.c_cam, cfg.c_rad, cfg.c_fuse)
            else:
                raise ValueError(cfg.fusion)
            if cfg.trunk_kind == "second":
                self.trunk = SecondTrunk(cin=cfg.c_fuse)
                c = self.trunk.out_channels
            elif cfg.trunk_kind == "dualbev":
                self.trunk = BEVTrunk(cin=cfg.c_fuse, width=cfg.trunk_width, detail=cfg.trunk_detail,
                                      fpn_original_order=cfg.trunk_fpn_original)
                if cfg.trunk_pretrained:                              # detection-trained BEV encoder of DualBEV (SOA init)
                    rep = self.trunk.load_pretrained(cfg.trunk_pretrained)
                    if rep["missing"] or rep["shape_mismatch"]:
                        raise RuntimeError(f"trunk weights do not fit: {rep['missing'][:3]} {rep['shape_mismatch'][:3]}")
                c = self.trunk.img_bev_encoder_neck.out_channels + cfg.trunk_detail
            else:
                raise ValueError(cfg.trunk_kind)
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
        self.sem_head = TimedHead(SemanticHead(c, len(self.spec)), c, self.mover)
        self.obj_head = TimedHead(ObjectHead(c, groups=NUSC_GROUPS, direction=cfg.object_direction), c, self.mover)
        if cfg.refine:
            import random
            self.refiner = BoxRefiner(c, groups=NUSC_GROUPS, use_dt=True)
            self._rng = random.Random(0)                          # candidate jitter
        if cfg.aux_heads:                                           # the layouts BEVLoss expects for them
            self.aux_cam = SemanticHead(cfg.c_cam, len(layer_spec(False, True)))
            self.aux_rad = SemanticHead(cfg.c_rad, len(radar_aux_spec(cfg.radar_stationary)))
        self.c_trunk = c

    # ---- camera
    def camera_bevs(self, batch):
        """(B, F, c_cam, X, Y) camera BEV per frame."""
        img = batch["img"]
        B, Fn = img.shape[:2]
        if self.cfg.camera_off:
            return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=img.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.freeze_camera):
            prob, ctx = self.camera.image_features(img.flatten(0, 1), batch["cam_vec"].flatten(0, 1))
        prob, ctx = prob.view(B, Fn, *prob.shape[1:]), ctx.view(B, Fn, *ctx.shape[1:])
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
        hw = batch["hw"] if "hw" in batch else batch["img"].shape[-2:]
        vis = [camera_visibility(self.grid, self.cfg.z_levels, self.lift.depth_centres, batch["K"][b, 0],
                                 batch["T_cam_car"][b, 0], hw, batch.get("ida"), self.cfg.oversample)
               for b in range(batch["K"].shape[0])]
        return torch.stack(vis)[:, None].to(like)

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
        p = res["out"][float(dt)]
        dets = decode({kk: v.float().cpu() for kk, v in p["obj"].items()}, self.grid, NUSC_GROUPS, k=k,
                      threshold=threshold)
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
        cam_frames = self.camera_bevs(batch)
        steps = [[tuple(torch.as_tensor(t, dtype=cam_frames.dtype, device=cam_frames.device) for t in s) for s in f]
                 for f in batch["steps"]]
        cam_obs = self.camera_visibility(batch, cam_frames) if isinstance(self.cam_temporal, TemporalFusion) else None
        flip = batch.get("flip")                                 # mirror augmentation (training/augment.py): the other inputs
        if flip is not None and bool(flip.any()):                # and the targets of these samples were mirrored by flip_batch
            cam_frames, cam_obs = flip_cam(cam_frames, cam_obs, flip)
        cam, cam_state = self._branch_temporal(self.cam_temporal, cam_frames, steps, state.get("cam"), step, cam_obs)
        rad = (torch.zeros(cam.shape[0], self.cfg.c_rad, self.grid.nx, self.grid.ny, device=cam.device)
               if self.cfg.radar_kind == "off" else self.radar(batch["radar"], cam.shape[0]))
        rad, rad_state = self._branch_temporal(self.rad_temporal, rad[:, None], [[]], state.get("rad"), step)
        S = self.trunk.features(cam, rad) if self.fusion is None else self.trunk.features(self.fusion(cam, rad))
        if self.head_reduce is not None:
            S = self.head_reduce(S)
        v, w = batch["v"].to(S), batch["omega"].to(S)
        out = {}
        refine_gt = (sample_targets(batch["gt"]) if self.cfg.refine and self.training and batch.get("gt") else {})
        for dt in dts:
            e = (ego or {}).get(float(dt))
            dt_i, v_i, w_i = (dt, v, w) if e is None else (torch.as_tensor(x).to(S) for x in e)
            sem, a = self.sem_head(S, dt_i, v_i, w_i, return_aux=True)
            obj, ao = self.obj_head(S, dt_i, v_i, w_i, return_aux=True)
            p = {"sem": sem, "obj": obj, "mass": a["mass"]}
            if self.cfg.refine:
                p["obj_feat"] = ao["state"]                          # what the refiner samples (also at run time)
                p["obj_dt"] = torch.as_tensor(dt_i).to(S).reshape(-1).expand(S.shape[0])
                if len(refine_gt.get(float(dt), [])) == S.shape[0]:
                    p["refine"] = self._refine_train(obj, ao["state"], refine_gt[float(dt)], dt_i)
            if abs(float(dt)) < 1e-6:
                p["v_hat"] = self.mover.motion(S)
                if self.cfg.aux_heads and self.training:
                    p["cam_aux"], p["radar_aux"] = self.aux_cam(cam), self.aux_rad(rad)
            out[float(dt)] = p
        return {"out": out, "state": {"cam": cam_state, "rad": rad_state}, "features": S}


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
             {"radar_kind": "pillars"}, {"radar_kind": "off"}, {"trunk_kind": "second", "head_channels": 96},
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
