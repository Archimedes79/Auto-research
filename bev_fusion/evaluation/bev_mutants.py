"""Mutation check of the response test: plant ONE deliberate bug at a time into the metrics (a monkeypatch, no file is edited) and count how many rows of
evaluation/bev_response.py go OFF. A mutant with 0 rows OFF ESCAPED: the test could not have found that bug. Used to find the three weak spots of the first
version of the response test (AVE as L1, TP threshold, AOE wrap), which now have their own rows.

    python evaluation/bev_mutants.py [n_frames] [name-part ...]        (from bev_fusion, CPU, about 8 min for all 12 at 24 frames; a name part selects mutants)

Mutants: ROC-AUC reported as 1 - AUC; ECE without bin weights; Brier without the + P term; RMSE without the square root; ASE = 1 - sqrt(IoU); AVE as L1 instead
of L2; ATE x 1.1; front window to 70 m; TP threshold 1 m instead of 2 m; AOE without the 2 pi wrap; moving-cell velocity sums = all cells; camera mask = every cell
(patched on the code side only: the test derives its cell sets from the same `masks()` function, so a bug INSIDE masks() would hit both sides and escape - the masks
are checked by `bev_gtcheck.py` against the poses and by eye, not by this test)."""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT]
import numpy as np  # noqa: E402

import evaluation.bev_response as R  # noqa: E402  (sets the import paths)
import evaluation.bev_curves as C  # noqa: E402
import evaluation.metrics as M  # noqa: E402
import evaluation.bev_fusion_area as FA  # noqa: E402
from evalsuite.comparison import boxes as B  # noqa: E402

orig = {"care": FA.CARE, "near": FA.NEAR, "prepare": FA.prepare, "share": FA.footprint_share, "farea": FA.fusion_area, "cls": C.classification_numbers, "reg": C.regression_numbers, "tp": B.tp_errors, "front": M.in_front, "thr": M.TP_THRESHOLD, "vs": C.velocity_sums}


def m_auc_inverted():
    C.classification_numbers = lambda h: dict(orig["cls"](h), roc_auc=1 - orig["cls"](h)["roc_auc"])


def m_ece_unweighted():
    def f(h):
        out = orig["cls"](h)
        n_bin = h[0] + h[1]
        k = n_bin > 0
        out["ece"] = float(np.mean(np.abs(h[0][k] / n_bin[k] - h[2][k] / n_bin[k])))
        return out
    C.classification_numbers = f


def m_brier_without_P():
    def f(h):
        out = orig["cls"](h)
        out["brier"] = float((h[3].sum() - 2.0 * h[4].sum()) / (h[0].sum() + h[1].sum()))
        return out
    C.classification_numbers = f


def m_rmse_no_sqrt():
    C.regression_numbers = lambda s: dict(orig["reg"](s), rmse=orig["reg"](s)["rmse"] ** 2)


def m_ase_sqrt():
    B.tp_errors = lambda d, g: dict(orig["tp"](d, g), ase=1 - math.sqrt(max(0.0, 1 - orig["tp"](d, g)["ase"])))


def m_ave_L1():
    B.tp_errors = lambda d, g: dict(orig["tp"](d, g), ave=abs(d["vx"] - g["vx"]) + abs(d["vy"] - g["vy"]))


def m_ate_scaled():
    B.tp_errors = lambda d, g: dict(orig["tp"](d, g), ate=1.1 * orig["tp"](d, g)["ate"])


def m_window_70m():
    M.in_front = lambda b, x_max=60.0, half_fov=M.FRONT_HALF_FOV: orig["front"](b, 70.0, half_fov)


def m_tp_threshold_1m():
    M.TP_THRESHOLD = 1.0


def m_aoe_no_wrap():
    B.tp_errors = lambda d, g: dict(orig["tp"](d, g), aoe=abs(d["yaw"] - g["yaw"]) if "yaw" in d and "yaw" in g else float("nan"))


def m_velocity_sums_skip_moving():
    def f(v, gv, cells):
        out = orig["vs"](v, gv, cells)
        out[1] = out[0]
        return out
    C.velocity_sums = f


def m_fov_mask_all():
    import evaluation.bev_task as bt
    real = bt.masks
    bt.masks = lambda g: dict(real(g), fov=np.ones_like(real(g)["fov"]))
    C.masks = bt.masks


def m_care_threshold_03():                                           # the response rows count the classes with the same function: the module's own selftest catches this one
    FA.CARE = 0.3


def _near_neutral(d, gts_all, scored):
    return any(b["group"] == d["group"] and FA._dist(d, b) <= FA.NEAR for b in gts_all if b not in scored)


def m_neutral_rule_off():                                            # a detection on a neutral (partial / don't-care) box that lies inside the area counts again
    real = FA.prepare

    def prepare(dets, gts_all, shares, area, include_partial):
        keep, scored = real(dets, gts_all, shares, area, include_partial)
        return keep + [d for d in dets if d not in keep and orig["share"](d, area) >= 0.5], scored
    FA.prepare = prepare


def m_area_rule_off():                                               # detections outside the fusion area are kept (those on a neutral box stay out: only the area rule is removed)
    real = FA.prepare

    def prepare(dets, gts_all, shares, area, include_partial):
        keep, scored = real(dets, gts_all, shares, area, include_partial)
        return keep + [d for d in dets if d not in keep and not _near_neutral(d, gts_all, scored)], scored
    FA.prepare = prepare


def m_fusion_area_ignores_hidden():
    FA.fusion_area = lambda g: (np.asarray(g["camera"], np.float32) > 0) & (np.asarray(g["radar"], np.float32) > 0)


MUTANTS = [("ROC-AUC reported as 1 - AUC", m_auc_inverted), ("ECE without bin weights", m_ece_unweighted), ("Brier without the + P term", m_brier_without_P),
           ("RMSE without the square root", m_rmse_no_sqrt), ("ASE = 1 - sqrt(IoU)", m_ase_sqrt), ("AVE as L1 instead of L2", m_ave_L1), ("ATE x 1.1", m_ate_scaled),
           ("front window to 70 m", m_window_70m), ("TP threshold 1 m instead of 2 m", m_tp_threshold_1m), ("AOE without the 2 pi wrap", m_aoe_no_wrap),
           ("moving-cell velocity sums = all cells", m_velocity_sums_skip_moving), ("camera mask = every cell (code side only)", m_fov_mask_all),
           ("care threshold 0.3 instead of 0.5", m_care_threshold_03), ("neutral rule off (detections on neutral boxes count)", m_neutral_rule_off),
           ("detections outside the fusion area are kept", m_area_rule_off), ("fusion area ignores hidden cells", m_fusion_area_ignores_hidden)]


def restore():
    C.classification_numbers, C.regression_numbers, B.tp_errors, M.in_front, M.TP_THRESHOLD, C.velocity_sums = (
        orig["cls"], orig["reg"], orig["tp"], orig["front"], orig["thr"], orig["vs"])
    FA.CARE, FA.NEAR, FA.footprint_share, FA.fusion_area, FA.prepare = orig["care"], orig["near"], orig["share"], orig["farea"], orig["prepare"]


def off(rows):
    return [r for r in rows if not (abs(r[2] - r[3]) <= r[4] or (math.isnan(r[2]) and math.isnan(r[3])))]


def main(argv):
    n = int(argv[0]) if argv and argv[0].isdigit() else 24
    only = [a for a in argv if not a.isdigit()]
    sig = R.G.resolve_sig(None)
    _, base = R.all_rows(n, sig, 0)
    print(f"baseline: {len(base)} rows, {len(off(base))} OFF")
    escaped = []
    for name, apply in MUTANTS:
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        restore()
        apply()
        try:
            _, rows = R.all_rows(n, sig, 0)
            bad = off(rows)
            if not bad:                                                     # the unit selftest of the fusion-area module is the second net
                try:
                    FA.selftest()
                except AssertionError as e:
                    bad = [("bev_fusion_area selftest", str(e)[:40], 0.0, 1.0, 0.0)]
            print(f"{'CAUGHT ' if bad else 'ESCAPED'}  {name:55s} {len(bad):3d} rows off" + (f"   e.g. {bad[0][0][:40]} / {bad[0][1]}" if bad else ""), flush=True)
            if not bad:
                escaped.append(name)
        except Exception as e:                                              # a crash is a catch too
            print(f"CAUGHT   {name:55s} (raised {type(e).__name__}: {str(e)[:60]})", flush=True)
    restore()
    print("escaped:", escaped)
    return 1 if escaped else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
