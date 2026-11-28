"""The model that reads cached features of the frozen camera heads instead of running them (the training and evaluation form of BEVFusionModel)."""
import torch
import torch.utils.checkpoint

from bev_fusion.net.prep.augment import jitter_images
from bev_fusion.net.model.lift_pd import lift_key


def CachedBEVFusionModel(cfg=None):
    """BEVFusionModel whose camera branch reads batch["prob"], batch["ctx"] (the frozen heads are not instantiated)."""
    from bev_fusion.net.model.model import CAMERAS, BEVFusionModel, ModelConfig
    cfg = cfg or ModelConfig()

    class _Cached(BEVFusionModel):
        CACHED = True

        def _student_images(self, batch, Fn):
            """camera_student (t22, t26): the images the student reads -> (current frame (B, 3, 256, 704), older frames (B, F - 1, 3, 256, 704) or None). The
            cached uint8 JPEGs batch["lane_img"] and, with
            student_frames 'all' and F > 1, batch["lane_img_old"] (frames 1 .. F - 1 in the order of the batch); a one-frame batch (a stream step) needs no
            older image. In training mode a
            batch["img_jitter"] (the trainer's run_clip) makes them the photometrically jittered floats (augment.jitter_images): the student's input only, the
            cached teacher outputs are those of the clean image"""
            if "lane_img" not in batch:
                raise KeyError("camera_student reads the current key-frame image batch['lane_img']: set TrainConfig.lane_images (the folder of the key-frame JPEGs, data/cache/img_front)")
            cur, old = batch["lane_img"], None
            if self.cfg.student_frames == "all" and Fn > 1:
                if "lane_img_old" not in batch:
                    raise KeyError("student_frames='all' reads the older frames' images batch['lane_img_old']: set TrainConfig.lane_images and have its sweeps sub-folder "
                                   "(python -P -m bev_fusion.training.build_image_cache build-cache-older <config>)")
                old = batch["lane_img_old"]
                if old.shape[1] != Fn - 1:
                    raise ValueError(f"student_frames='all': batch['lane_img_old'] holds {old.shape[1]} older frame(s), the batch has {Fn - 1}")
            if self.training and batch.get("img_jitter") is not None:
                cur, old = jitter_images(cur, batch["img_jitter"], 0), None if old is None else jitter_images(old, batch["img_jitter"], 1)
            return cur, old

        def _run_student(self, img):
            """camera_student: the student on the current frame's images (B, 3, 256, 704) -> its output dict; no graph unless it trains"""
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.cfg.student_trainable):
                return self.student(img)

        def _run_student_older(self, img):
            """student_frames 'all': the student on the images of the OLDER frames (B, F - 1, 3, 256, 704) in ONE batched forward, never with a graph (BatchNorm
            of the student is held in eval mode, so
            the batch of frames does not matter) -> prob / ctx of each (B, F - 1, ...). The older frames enter as constants: the student's own gradient comes
            from the current frame. In stage B2 the
            older frame was made one step before by the same student in the stream; here every frame of the window is made by it, once per call. Values, not
            gradients: in TRAINING mode a loss
            on the older frames' BEV still reaches the student through the batch statistics of the BatchNorm layers behind the lift (cam_reduce, cam_merge,
            cam_temporal) that frame 0 is
            normalised with, exactly as the cached frames are coupled to frame 0 with 'current'"""
            B, K = img.shape[:2]
            with torch.no_grad():
                out = self.student(img.flatten(0, 1))
            return {k: v.reshape(B, K, *v.shape[1:]) for k, v in out.items() if k.startswith(("prob", "ctx"))}

        def camera_bevs(self, batch):
            prob = batch["prob"].float()
            B, Fn = prob.shape[:2]
            if self.cfg.camera_off:                                     # radar-only system (was ignored here until 2026-10-30:
                return torch.zeros(B, Fn, self.cfg.c_cam, self.grid.nx, self.grid.ny, device=prob.device)   # t2_cam_off ran WITH camera)
            stud, old, side = None, None, {}
            if self.student is not None:
                cur_img, old_img = self._student_images(batch, Fn)
                stud = self._run_student(cur_img)
                # the older frames AFTER the current one: under autocast the weight casts made inside no_grad would otherwise be reused by the call with a graph
                # (the student would get no gradient)
                old = None if old_img is None else self._run_student_older(old_img)
            recompute_lift = stud is not None and self.cfg.student_trainable and self.training and torch.is_grad_enabled()
            bevs = []
            for i, src in enumerate(CAMERAS[self.cfg.camera_features]):  # one lift + collapse per feature source, each with its own ida
                s = src.suffix
                p, ctx = (prob if i == 0 else batch["prob" + s].float()), batch["ctx" + s].float()
                # t22: frame 0 from the student (`side` for the anchor loss: frame 0 only), the older frames keep their cached teacher outputs
                if stud is not None:
                    h = self.student.HEAD_OF[src.features]              # or, with student_frames 'all' (t26), come from the student too, as constants
                    side.update({"p" + h: stud["prob" + h], "c" + h: stud["ctx" + h], "l" + h: stud["logit" + h], "t_p" + h: p[:, 0].detach(), "t_c" + h: ctx[:, 0].detach()})
                    p = torch.cat([stud["prob" + h].float()[:, None], p[:, 1:] if old is None else old["prob" + h].float()], 1)
                    ctx = torch.cat([stud["ctx" + h].float()[:, None], ctx[:, 1:] if old is None else old["ctx" + h].float()], 1)
                lift, collapse = getattr(self, "lift" + s), getattr(self, "cam_reduce" + s)
                vols = []
                for b in range(B):                                      # one calibration per sample (scenes differ)
                    args = (ctx[b], p[b], batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida" + s])
                    # t22: the student's depth + context carry a graph into the lift; without this the 118 x C frustum features of the 9 frames of a micro-batch
                    # stay in memory
                    if recompute_lift:
                        key = lift_key(*args[2:])       # the projection-table key, made once here: no recomputation inside a backward has to wait for the GPU
                        vols.append(collapse(torch.utils.checkpoint.checkpoint(lift, *args, key=key, use_reentrant=False)))
                    else:
                        vols.append(collapse(lift(*args)))
                bevs.append(torch.stack(vols))
            self._student_side = side or None
            if len(bevs) == 1:
                return bevs[0]
            return self.cam_merge(torch.cat(bevs, 2).flatten(0, 1)).view_as(bevs[0])

    import dataclasses
    m = _Cached(dataclasses.replace(cfg, pretrained=""))                  # no checkpoint: the heads are dropped below
    del m.camera
    return m
