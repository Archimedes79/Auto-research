"""Mutation check of the response test: plant one deliberate bug at a time into the metrics (a monkeypatch, no file is edited) and count how many rows of
evaluation/bev_response.py go OFF. A mutant with 0 rows OFF ESCAPED: the test could not have found that bug.
Run from bev_fusion (WSL, gcar python): python temp/pitch/mutants.py [n_frames] [only-name ...]"""
import math
import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

import evaluation.bev_response as R  # noqa: E402  (sets the import paths)
import evaluation.bev_curves as C  # noqa: E402
import evaluation.metrics as M  # noqa: E402
from evalsuite.comparison import boxes as B  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 24
ONLY = [a for a in sys.argv[1:] if not a.isdigit()]
orig = {"cls": C.classification_numbers, "reg": C.regression_numbers, "tp": B.tp_errors, "front": M.in_front, "thr": M.TP_THRESHOLD}


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
        out = C.__dict__["_orig_vs"](v, gv, cells)
        out[1] = out[0]
        return out
    C._orig_vs = C.velocity_sums
    C.velocity_sums = f


def m_fov_mask_all():                                   # a wrong MASK: the test derives its cell sets from the same masks() function, so it cannot see this one
    import evaluation.bev_curves as c2
    import evaluation.bev_task as bt
    real = bt.masks
    bt.masks = lambda g: dict(real(g), fov=np.ones_like(real(g)["fov"]))
    c2.masks = bt.masks


MUTANTS = [("ROC-AUC reported as 1 - AUC", m_auc_inverted), ("ECE without bin weights", m_ece_unweighted), ("Brier without the + P term", m_brier_without_P),
           ("RMSE without the square root", m_rmse_no_sqrt), ("ASE = 1 - sqrt(IoU)", m_ase_sqrt), ("AVE as L1 instead of L2", m_ave_L1), ("ATE x 1.1", m_ate_scaled),
           ("front window to 70 m", m_window_70m), ("TP threshold 1 m instead of 2 m", m_tp_threshold_1m), ("AOE without the 2 pi wrap", m_aoe_no_wrap),
           ("moving-cell velocity sums = all cells", m_velocity_sums_skip_moving), ("camera mask = every cell (shared by test and code)", m_fov_mask_all)]


def restore():
    C.classification_numbers, C.regression_numbers, B.tp_errors, M.in_front, M.TP_THRESHOLD = orig["cls"], orig["reg"], orig["tp"], orig["front"], orig["thr"]
    if hasattr(C, "_orig_vs"):
        C.velocity_sums = C._orig_vs


def off(rows):
    return [r for r in rows if not (abs(r[2] - r[3]) <= r[4] or (math.isnan(r[2]) and math.isnan(r[3])))]


samples, base = R.all_rows(N, "15efbde2d1", 0)
print(f"baseline: {len(base)} rows, {len(off(base))} OFF")
escaped = []
for name, apply in MUTANTS:
    if ONLY and not any(o.lower() in name.lower() for o in ONLY):
        continue
    restore()
    apply()
    try:
        _, rows = R.all_rows(N, "15efbde2d1", 0)
        bad = off(rows)
        print(f"{'CAUGHT ' if bad else 'ESCAPED'}  {name:55s} {len(bad):3d} rows off" + (f"   e.g. {bad[0][0][:40]} / {bad[0][1]}" if bad else ""), flush=True)
        if not bad:
            escaped.append(name)
    except Exception as e:
        print(f"CAUGHT   {name:55s} (raised {type(e).__name__}: {str(e)[:60]})", flush=True)
restore()
print("escaped:", escaped)
