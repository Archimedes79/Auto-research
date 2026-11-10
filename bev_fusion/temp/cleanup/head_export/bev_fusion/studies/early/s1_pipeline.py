"""Gate of the camera-only S1 pipeline on synthetic data (experiments/s1a_lift_pd_vs_p, section 3; plan.md 10.1): the REAL
module (training/legacy/s1_camera.py), the REAL driver (training/legacy/train_s1.py: loss, training loop, per-scene SegMeters, null baseline,
paired scene bootstrap) trained and tested on artificial data with a known answer (studies/early/synthetic_s1_data.py: the real
cache format at the real feature resolution, stride 16). What it proves: the pipeline is wired (it can overfit), it learns
(above the prior-map null), the depth weight helps exactly when the depth carries information, and the presets of the
published designs (DualBEV HeightTrans, Simple-BEV, LSS) run through the same code and learn the same task - so a later
real-data difference between arms comes from the data and the heads, not from the harness, and our design is anchored to
the state of the art, not free-floating.

GATES (written before the first run, 2026-10-26; class mean = mean of IoU@0.5 m over lanes / vehicles / pedestrians on
camera-visible cells, held-out synthetic scenes):
  G1 wiring     overfit ONE scene (ours, oracle depth): every class IoU@0.5 m >= 0.9 on that scene
  G2 learning   ours + PD + oracle depth: class mean >= null + 0.15 on both seeds
  G3 depth      vehicles: PD-oracle - P >= +0.05 (mean of two seeds each)
  G4b control   PD with pixel-permuted depth (the sample's own histograms at random other pixel groups): vehicles <= P + 0.03 (the depth
                CONTENT at its place carries the gain, not the extra weight). History: G4 as first written (PD with the NEXT sample's depth, vehicles <=
                P + 0.03) FAILED in the first full run of 2026-10-31 (0.477 vs 0.286): that depth keeps the scene-independent flat-ground depth (89 % of
                ground groups within 1 m of the truth), so it is no null; replaced by G4b, not removed silently
  G5 anchors    each published preset (simplebev, lss, dualbev) learns: class mean >= null + 0.15; and ours >= max of them - 0.02
Printed besides the gates: per class and range band, paired scene bootstrap PD - P, null, parameters, seconds per run.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python studies/early/s1_pipeline.py [--steps N] [--arms a,b] [--quick]
(GPU, about 1 s per 10 steps for width 0.5 on a shared 3060; the default 1200 steps x 9 runs takes about 1.5 hours with the DualBEV width-1.0 trunk.)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))                      # the bev_fusion folder
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "training", "legacy"))                # s1_camera, train_s1 (off the chain since 2026-10-31)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import synthetic_s1_data as D  # noqa: E402
import train_s1 as T  # noqa: E402
from s1_camera import CameraBEVSeg, S1Config, n_params  # noqa: E402

NAMES = D.CLASS_NAMES
RESULT = os.path.join(ROOT, "temp", "checks", "s1_pipeline_result.json")
# arm name -> (preset, overrides, depth variant, seeds)
ARMS = {"ours PD oracle": ("ours", {}, "oracle", (0, 1)),
        "ours P": ("ours", {"lift": "p"}, "oracle", (0, 1)),
        "ours PD blurred": ("ours", {}, "blurred", (0,)),
        "ours PD shuffled": ("ours", {}, "shuffled", (0,)),
        "ours PD permuted": ("ours", {}, "permuted", (0,)),
        "simplebev (P, 8 slices)": ("simplebev", {}, "oracle", (0,)),
        "lss (S, heights summed)": ("lss", {}, "oracle", (0,)),
        "dualbev (13 heights summed, width 1.0)": ("dualbev", {}, "oracle", (0,))}


def run_arm(name, spec, seed, data, steps, device, log):
    preset, over, kind, _ = spec
    cfg = S1Config.preset(preset, **over)
    model = CameraBEVSeg(cfg)
    mk = lambda idx: D.batch([data["train"][i] for i in idx], data, kind)       # noqa: E731
    mv = lambda idx: D.batch([data["val"][i] for i in idx], data, kind)         # noqa: E731
    t0 = time.time()
    final = T.train(model, mk, len(data["train"]), steps, bs=8, seed=seed, device=device, log=log)
    meters = T.evaluate(model, mv, list(range(len(data["val"]))), NAMES, model.grid, device)
    return meters, {"final_loss": final, "seconds": time.time() - t0, "params_M": n_params(model) / 1e6}


def table_row(summ):
    cols = []
    for n in NAMES:
        s = summ[n]
        cols.append(f"{s['all'][1]:.3f} ({s['0-20'][1]:.2f}/{s['20-40'][1]:.2f}/{s['40-60'][1]:.2f})")
    return " | ".join(cols) + f" | {T.class_mean(summ, NAMES):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--arms", default="")
    ap.add_argument("--quick", action="store_true", help="300 steps, one seed, 3 arms: a smoke run, no gate")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda s: print(s, flush=True)                                          # noqa: E731
    # 0 self-tests of the parts
    D._selftest()
    T._selftest()
    from s1_camera import _selftest as s1_selftest
    s1_selftest()
    data = D.load()
    grid = CameraBEVSeg(S1Config()).grid
    n_tr, n_va = len(data["train"]), len(data["val"])
    log(f"data: {n_tr} train / {n_va} held-out synthetic scenes, device {device}")
    # null baseline (prior map)
    tr_gt = (torch.stack([s["gt"] for s in data["train"]]).float() > torch.tensor(D.GT_THRESHOLDS).view(1, -1, 1, 1)).float()
    tr_mask = torch.stack([s["mask"] for s in data["train"]]).float()[:, None]
    mv = lambda idx: D.batch([data["val"][i] for i in idx], data)                # noqa: E731
    null, thr = T.null_meters(tr_gt, tr_mask, mv, list(range(n_va)), NAMES, grid)
    null_s = T.summary(null, NAMES)
    log(f"null (prior map, thresholds {thr}): " + table_row(null_s))
    # G1 wiring: overfit one scene
    one = lambda idx: D.batch([data["train"][0]] * len(idx), data)               # noqa: E731
    m1 = CameraBEVSeg(S1Config())
    steps1 = 60 if a.quick else 250
    T.train(m1, one, 1, steps1, bs=4, lr=5e-4, seed=0, device=device, warmup=20, log_every=0)
    meters1 = T.evaluate(m1, one, [0], NAMES, m1.grid, device)
    s1 = T.summary(meters1, NAMES)
    g1 = {n: s1[n]["all"][1] for n in NAMES}
    log("G1 overfit one scene, IoU@0.5 m: " + ", ".join(f"{n} {v:.3f}" for n, v in g1.items()))
    del m1
    # G2-G5 learning arms
    arms = {k: v for k, v in ARMS.items() if not a.arms or any(x in k for x in a.arms.split(","))}
    if a.quick:
        arms = {k: (v[0], v[1], v[2], v[3][:1]) for k, v in list(ARMS.items())[:2] + list(ARMS.items())[4:5]}
    steps = 300 if a.quick else a.steps
    res, meters, info = {}, {}, {}
    for name, spec in arms.items():
        for seed in spec[3]:
            log(f"arm {name}, seed {seed}, {steps} steps")
            m, inf = run_arm(name, spec, seed, data, steps, device, log)
            meters[(name, seed)], info[(name, seed)] = m, inf
            res[(name, seed)] = T.summary(m, NAMES)
            log(f"  -> {table_row(res[(name, seed)])} | {inf['seconds']:.0f} s, {inf['params_M']:.1f} M parameters")
            os.makedirs(os.path.dirname(RESULT), exist_ok=True)
            json.dump({"steps": steps, "null": null_s, "G1": g1,
                       "runs": {f"{k[0]} s{k[1]}": {"iou": v, **info[k]} for k, v in res.items()}},
                      open(RESULT, "w"), indent=1, default=float)
    # report
    log(f"\nheld-out synthetic scenes ({n_va}); IoU@0.5 m all (0-20 / 20-40 / 40-60 m) per class: "
        + " | ".join(NAMES) + " | class mean")
    log(f"{'null (prior map)':42s} | " + table_row(null_s))
    for (name, seed), s in res.items():
        log(f"{name + ' s' + str(seed):42s} | " + table_row(s))
    if a.quick:
        log("quick run: no gates")
        return 0

    def mean_over_seeds(name, f):
        v = [f(res[(name, s)]) for s in ARMS[name][3] if (name, s) in res]
        return float(np.mean(v)), v
    cm = lambda s: T.class_mean(s, NAMES)                                          # noqa: E731
    veh = lambda s: s["vehicles"]["all"][1]                                        # noqa: E731
    gates = {}
    gates["G1 wiring: every class >= 0.9 on the overfit scene"] = min(g1.values()) >= 0.9
    nul = cm(null_s)
    _, pd_cm = mean_over_seeds("ours PD oracle", cm)
    gates["G2 learning: ours PD oracle >= null + 0.15 on both seeds"] = min(pd_cm) >= nul + 0.15
    pd_v, _ = mean_over_seeds("ours PD oracle", veh)
    p_v, _ = mean_over_seeds("ours P", veh)
    gates[f"G3 depth: vehicles PD - P = {pd_v - p_v:+.3f} >= +0.05"] = pd_v - p_v >= 0.05
    sh_v, _ = mean_over_seeds("ours PD shuffled", veh)
    log(f"info: PD with the NEXT sample's depth (shuffled) reaches vehicles {sh_v:.3f} against P {p_v:.3f}: it keeps the scene-independent ground depth")
    if ("ours PD permuted", 0) in res:
        pm_v, _ = mean_over_seeds("ours PD permuted", veh)
        gates[f"G4b control: vehicles with pixel-permuted depth {pm_v:.3f} <= P {p_v:.3f} + 0.03"] = pm_v <= p_v + 0.03
    soa = {k: cm(res[(k, 0)]) for k in ARMS if k.startswith(("simplebev", "lss", "dualbev"))}
    gates["G5a anchors learn: " + ", ".join(f"{k.split(' ')[0]} {v:.3f}" for k, v in soa.items()) + f" >= null {nul:.3f} + 0.15"] = \
        all(v >= nul + 0.15 for v in soa.values())
    ours = float(np.mean(pd_cm))
    gates[f"G5b ours {ours:.3f} >= best anchor {max(soa.values()):.3f} - 0.02"] = ours >= max(soa.values()) - 0.02
    # paired bootstrap PD - P (seed 0 pair, same scenes)
    for layer in NAMES:
        p, lo, hi = T.paired_bootstrap(meters[("ours P", 0)], meters[("ours PD oracle", 0)], layer)
        log(f"paired scene bootstrap PD - P, {layer}, IoU@0.5 m (seed 0, {n_va} scenes): {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    seed_sd = float(np.std([cm(res[("ours PD oracle", s)]) for s in (0, 1)]))
    log(f"seed spread of ours PD oracle (class mean over two seeds): sd {seed_sd:.3f}")
    ok = True
    for k, v in gates.items():
        log(f"{'PASS' if v else 'FAIL'}  {k}")
        ok &= bool(v)
    log(f"GATE S1 pipeline (synthetic): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
