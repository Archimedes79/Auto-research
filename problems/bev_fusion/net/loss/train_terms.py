"""The terms a training step adds to the total of BEVLoss, each with its weight over the steps; none of them exists at inference:

    consistency      the cross-time consistency of the predictions at dt = 0 and dt = +-0.5 s (loss_extras.cross_time_consistency), after a hold and a ramp
    radar_aux        the reconstruction side loss of the delta-slot radar encoder (radar_loss.py), decaying to a floor
    student_anchor   the distillation anchor of a trainable camera student (student_loss.py), falling to a floor

`add` puts them into the loss dict of one batch (chain.forward_loss calls it); in validation it logs what stands in their place and leaves the total alone."""
import torch

from net.loss.loss_extras import cross_time_consistency
from net.loss.losses import build_layer_targets
from net.loss.radar_loss import aux_weight, slot_aux_loss
from net.loss.student_loss import anchor_terms, anchor_weight


def consistency_weight(cfg, step_frac):
    """cfg.consistency x a linear ramp: 0 during the hold (first `hold` of the steps), 1 after hold + ramp."""
    hold, ramp = cfg.consistency_warmup
    return cfg.consistency * min(1.0, max(0.0, (step_frac - hold) / max(ramp, 1e-6)))


def consistency_term(model, loss, res, gts, gap_steps, device):
    """the cross-time consistency of the predictions at dt = 0 and dt = +-0.5 s: the layers whose kind is area / line / object, the visibility of the cells =
    camera and not
    occluded, the ego steps (batch["gap_steps"]) and the motion field of the batch. -> the result dict of cross_time_consistency."""
    sel = [i for i, (_, kind, _) in enumerate(loss.spec) if kind in ("area", "line", "object")]
    out = res["out"]
    probs = {dt: torch.sigmoid(o["sem"][:, sel].float()) for dt, o in out.items()}
    weights = {dt: build_layer_targets(gts[dt], loss.cfg, out[dt].get("mass"), dt, device)["w_cam"] for dt in out}
    steps = {dt: [g["prev" if dt < 0 else "next"] for g in gap_steps] for dt in out if dt != 0.0}
    return cross_time_consistency(probs, steps, out[0.0]["v_hat"].float(), model.grid, weights)


def add(out, model, loss, res, gts, batch, cfg, step_frac, device, training):
    """in place, the train-only terms of one batch: `out` is the loss dict of BEVLoss, `res` the model's result, `batch` the batch on the device. training: each
    term that
    is on is added to out["total"] with its weight and logged under its name; validation (training False): the total stays the task loss, the radar side logs
    the share of the
    detections its features hold (`radar_kept`), the student its distance to the teachers."""
    wc = consistency_weight(cfg, step_frac) if training and len(res["out"]) > 1 else 0.0
    if wc > 0:                                                                     # nothing is computed during the hold
        ct = consistency_term(model, loss, res, gts, batch["gap_steps"], device)
        out["consistency"] = ct["total"]
        out["total"] = out["total"] + wc * ct["total"]
    side = res.get("radar_side")                                                   # radar_kind "delta_slots" only
    if side is not None and cfg.radar_aux > 0:
        features, state = side["features"].float(), side["state"].float()
        if training:                                                               # the side loss: helps early, decays to a floor
            out["radar_aux"] = slot_aux_loss(model.radar, features, state)
            out["total"] = out["total"] + aux_weight(step_frac, cfg.radar_aux, cfg.radar_aux_floor, cfg.radar_aux_half) * out["radar_aux"]
        else:                                                                      # how much of the detections the features hold
            out["radar_kept"] = model.radar.kept_fraction(batch["radar"], features.shape[0], features=features)[0]
    side = res.get("student_side")                                                 # model.camera_student only
    if side is not None and cfg.student_anchor > 0:
        anchor = anchor_terms(side)["total"]
        if training:                                                               # the student's current-frame outputs stay near the teachers'
            out["student_anchor"] = anchor
            out["total"] = out["total"] + anchor_weight(step_frac, cfg.student_anchor, cfg.student_anchor_floor, cfg.student_anchor_until) * anchor
        else:                                                                      # how far the student has drifted from the teachers
            out["student_anchor"] = anchor.detach()
