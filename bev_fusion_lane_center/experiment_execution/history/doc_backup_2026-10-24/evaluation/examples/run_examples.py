"""Test on example images and with real numbers: champion D7 (fast path) against the control E94 on all 873 validation frames.
 1. both graph sets are scored with the shared metrics; the numbers are compared with the ones stored when the runs were made (does the evaluation reproduce history?)
 2. scene-bootstrap intervals and the paired difference (is S of the champion better than the control, beyond the noise band 0.010?)
 3. example images (ground truth | control | champion) of frames chosen by the per-frame difference, and a video of one scene
Needs the GPU (builder forward).   Run: tools/env/_py14.sh run_examples.py     Writes RESULT.md, result.json, val_examples.png, val_scene.mp4 next to this file.
"""
import json
import os
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import bootstrap as SB  # noqa: E402
import make_video as MV  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_timing as TM  # noqa: E402
import render as RD  # noqa: E402
import sparse_metrics as SM  # noqa: E402
import viz_graph as VG  # noqa: E402

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
HERE = f"{R}/evaluation/examples"
HOME = os.path.expanduser("~")
sp = json.load(open(f"{R}/data/splits.json"))

# ---------------------------------------------------------------- champion on all validation frames
net, cfg = TM.load_builder(288, 144)
ds = PD.BuilderData("val_clean", ks=None, train=False, with_prev=cfg.temporal)
ks = list(range(len(ds)))
raw = []
_, gts, _ = PI.predict(net, ds, ks, taus=(0.7,), bs=8, workers=2, raw=raw)
t0 = time.time()
champ = [PI.assemble(r, None, 0.7, cfg, fast=True, **TM.ASM) for r in raw]
asm_ms = (time.time() - t0) / len(raw) * 1e3
z = pickle.load(open(f"{R}/runs/e94_a/graphs_val_clean.pkl", "rb"))
ctrl = z["preds"]
assert len(ctrl) == len(champ) == len(gts) == 873 and list(z["ks"]) == ks, "frame sets differ"
scene = np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["scene"][np.array(sp["val"])]
assert len(scene) == len(gts)
print(f"{len(gts)} validation frames, {len(set(scene.tolist()))} scenes; assembly of the champion {asm_ms:.1f} ms per frame")

# ---------------------------------------------------------------- 1. scores and comparison with the stored history
rec_c, rec_k = SB.frame_records(champ, gts), SB.frame_records(ctrl, gts)
mc, mk = SB.pooled(rec_c), SB.pooled(rec_k)
chk = SM.card_any(champ, gts)
assert abs(chk["S"] - mc["S"]) < 1e-9 and abs(chk["F1@1.0"] - mc["F1@1.0"]) < 1e-9, "bootstrap.pooled != card_any"
hist = {}
for nm, p in (("control E94 (stored eval_val_clean)", f"{R}/runs/e94_a/eval_val_clean.json"), ("champion (stored finalA_val_clean)", f"{R}/runs/e123_ladder/cards/e123_D7_r3_f1.finalA_val_clean.json")):
    try:
        hist[nm] = json.load(open(p, encoding="utf-8-sig"))
    except Exception as e:
        hist[nm] = {"error": repr(e)[:80]}
print("stored control keys:", list(hist["control E94 (stored eval_val_clean)"].keys())[:12])
print("stored champion keys:", list(hist["champion (stored finalA_val_clean)"].keys())[:12])

# ---------------------------------------------------------------- 2. intervals and paired difference
iv_c, iv_k = SB.interval(rec_c, scene, B=500), SB.interval(rec_k, scene, B=500)
pd_ = SB.paired(rec_c, rec_k, scene, B=1000)

# ---------------------------------------------------------------- 3. example images
f1 = lambda r: 2 * r["rs"]["cnt"][1, 0] * r["rs"]["cnt"][1, 2] / max(r["rs"]["cnt"][1, 0] * r["rs"]["cnt"][1, 3] + r["rs"]["cnt"][1, 2] * r["rs"]["cnt"][1, 1], 1e-9)
fc, fk = np.array([f1(r) for r in rec_c]), np.array([f1(r) for r in rec_k])
d = fc - fk
order = np.argsort(d)
pick = list(order[-2:][::-1]) + list(order[:2]) + list(np.argsort(np.abs(d))[len(d) // 2 : len(d) // 2 + 2]) + list(np.argsort(fc + fk)[:2])
labels = ["champion much better", "champion much better", "control much better", "control much better", "typical (tie)", "typical (tie)", "hardest for both", "hardest for both"]
fig, axs = plt.subplots(len(pick), 3, figsize=(10.5, 3.4 * len(pick)))
for row, (i, lab) in enumerate(zip(pick, labels)):
    VG.panel(axs[row, 0], None, gts[i], title=f"frame {i} - ground truth ({lab})")
    VG.panel(axs[row, 1], ctrl[i], gts[i], title=f"control E94  F1@1 {fk[i]:.2f}")
    VG.panel(axs[row, 2], champ[i], gts[i], title=f"champion D7  F1@1 {fc[i]:.2f}")
fig.tight_layout()
fig.savefig(f"{HERE}/val_examples.png", dpi=70)
plt.close(fig)
# video: the scene with most frames, three panels side by side
sc = np.bincount(scene).argmax() if scene.min() >= 0 else scene[0]
fr = [i for i in range(len(scene)) if scene[i] == sc][:40]
arrs = []
for i in fr:
    a1 = RD.frame_array(None, gts[i], title=f"ground truth, frame {i}", size=4.0)
    a2 = RD.frame_array(ctrl[i], gts[i], title=f"control E94  F1@1 {fk[i]:.2f}", size=4.0)
    a3 = RD.frame_array(champ[i], gts[i], title=f"champion D7  F1@1 {fc[i]:.2f}", size=4.0)
    arrs.append(np.concatenate([a1, a2, a3], axis=1))
MV.frames_to_video(arrs, f"{HERE}/val_scene.mp4", fps=4)

# ---------------------------------------------------------------- write
out = dict(frames=len(gts), scenes=int(len(set(scene.tolist()))), assembly_ms_champion=asm_ms, champion=mc, control=mk, interval_champion=iv_c, interval_control=iv_k, paired_champion_minus_control=pd_)
json.dump(out, open(f"{HERE}/result.json", "w"), indent=1, default=float)
L = [f"# Example test: champion D7 vs control E94 on the validation set (generated by run_examples.py)", "",
     f"{len(gts)} frames of {out['scenes']} scenes, same frames and same metric code for both (`evaluation/sparse_lanes`). Champion = fast path, M 288 / K0 144, tau 0.7, assembly {asm_ms:.1f} ms per frame. Control = stored graphs of E94 (`runs/e94_a/graphs_val_clean.pkl`).", "",
     "| metric | control E94 (95 % over scenes) | champion D7 (95 % over scenes) | champion - control (95 %) | share of draws champion better |", "|---|---|---|---|---|"]
for k in SB.KEYS:
    if k in pd_:
        p, a_, b_ = pd_[k], iv_k.get(k), iv_c.get(k)
        L.append(f"| {k} | {a_['value']:.3f} [{a_['lo']:.3f}, {a_['hi']:.3f}] | {b_['value']:.3f} [{b_['lo']:.3f}, {b_['hi']:.3f}] | {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] | {p['p_a_better']:.2f} |")
L += ["", f"**Verdict on S (noise band 0.010):** {pd_['S']['verdict']}.", "",
      "Reading: Q = S without the sparsity term (rescaled; `sparse_metrics.quality`). S and F1 of the champion are compared with the control on identical frames; the interval comes from resampling whole scenes (frames of a scene are correlated). Vertices per 100 m and the length ratio belong to the sparsity argument (shorter, not longer graphs); J is the known weak spot.",
      "", "![examples](val_examples.png)", "", "Video of one scene: `val_scene.mp4` (ground truth | control | champion).", ""]
open(f"{HERE}/RESULT.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L[:14]))
