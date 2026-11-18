"""Schleife 5 / E71-E72: gepaarter Szenen-Bootstrap der wichtigsten Kontraste (je Szene Mittel ueber alle Bilder der Phase und die vier Ausfallarten; 21 TEST-Szenen, 5000 Ziehungen).
Aufruf (gcar): boot_e71.py -> out/loop2/e71_boot.json"""
import json
import pickle

import numpy as np

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
CONDS = ("dark", "cam_front3_off", "blur", "cam_front_off")


def per_scene(run, pol, sel):
    """Mittel des Querfehlers von pol ueber die Bilder, die sel(rec) erfuellen, je Szene und Ausfallart; dann Mittel ueber die Ausfallarten (gleiches Gewicht je Art wie in den Tabellen; Arten ohne Bilder in der Phase werden ausgelassen)"""
    per_cond = []
    for c in CONDS:
        rs = [sc for sc in pickle.load(open(f"{OUT}/{run}/frames_{c}.pkl", "rb")) if sc]
        row = []
        for i, sc in enumerate(rs):
            v = [r["lata"][pol] for r in sc if r["lata"] is not None and sel(r)]
            row.append(np.mean(v) if v else np.nan)
        per_cond.append(row)
    a = np.array(per_cond, float)
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=0)


def boot(a, b, n=5000, seed=0):
    d = a - b
    ok = np.isfinite(d)
    d = d[ok]
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), int(len(d))


PH = {
    "before": lambda r: r["phase"] == "before",
    "fail": lambda r: r["phase"] == "fail",
    "a0": lambda r: r["phase"] == "after" and 0 <= r["since_end"] < 3,
    "a1": lambda r: r["phase"] == "after" and 3 <= r["since_end"] < 6,
    "after": lambda r: r["phase"] == "after" and 0 <= r["since_end"] < 6,
}
# (Beschriftung, Lauf, Phase, Verfahren A, Verfahren B)  -> Differenz A - B (negativ: A hat den kleineren Querfehler)
CONTRASTS = [
    ("E71 no jump, during the failure: restart without gate - hold + restart + verify", "e71v_none_j3", "fail", "trackrestart", "holdverify"),
    ("E71 3 m jump before the failure, before the failure: hold + restart + verify - hold", "e71v_before_j3", "before", "holdverify", "hold"),
    ("E71 3 m jump before the failure, during the failure: hold + restart + verify - hold", "e71v_before_j3", "fail", "holdverify", "hold"),
    ("E71 3 m jump in the blind phase, 0-1.5 s after the return: hold + restart + verify - hold", "e71v_during_j3", "a0", "holdverify", "hold"),
    ("E71 3 m jump in the blind phase, 0-1.5 s after the return: hold + restart + verify - hold + restart", "e71v_during_j3", "a0", "holdverify", "holdrestart"),
    ("E71 3 m jump in the blind phase, 0-1.5 s after the return: hold + restart + verify - restart without gate", "e71v_during_j3", "a0", "holdverify", "trackrestart"),
    ("E71 6 m jump in the blind phase, 0-1.5 s after the return: hold + restart + verify - hold", "e71v_during_j6", "a0", "holdverify", "hold"),
    ("E71 6 m jump in the blind phase, 0-1.5 s after the return: hold + restart + verify - restart without gate", "e71v_during_j6", "a0", "holdverify", "trackrestart"),
    ("E72 no jump, during the bursts: restart without gate - hold + restart + verify", "e72_none_j3", "fail", "trackrestart", "holdverify"),
    ("E72 3 m jump in burst 2, after a burst (0-3 s): hold + restart + verify - hold", "e72_during_j3", "after", "holdverify", "hold"),
    ("E72 3 m jump in burst 2, after a burst (0-3 s): hold + restart + verify - restart without gate", "e72_during_j3", "after", "holdverify", "trackrestart"),
    ("E71 3 m jump in the blind phase, 0-1.5 s after the return: strict verify - hold + restart + verify", "e71v_during_j3", "a0", "holdstrict", "holdverify"),
    ("E71 3 m jump in the blind phase, 0-1.5 s after the return: strict verify - hold + restart", "e71v_during_j3", "a0", "holdstrict", "holdrestart"),
    ("E72 3 m jump in burst 2, after a burst (0-3 s): hold + restart + verify - hold + restart", "e72_during_j3", "after", "holdverify", "holdrestart"),
    ("E72 3 m jump in burst 2, after a burst (0-3 s): strict verify - hold + restart + verify", "e72_during_j3", "after", "holdstrict", "holdverify"),
    ("E72 3 m jump in burst 2, after a burst (0-3 s): strict verify - hold + restart", "e72_during_j3", "after", "holdstrict", "holdrestart"),
]


def main():
    res = []
    for lab, run, ph, a, b in CONTRASTS:
        A, B = per_scene(run, a, PH[ph]), per_scene(run, b, PH[ph])
        m, lo, hi, n = boot(A, B)
        res.append(dict(label=lab, run=run, phase=ph, a=a, b=b, mean=m, lo=lo, hi=hi, n_scenes=n))
        print(f"{lab}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={n})", flush=True)
    json.dump(res, open(f"{OUT}/e71_boot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
