"""E364 arm m (R6): the share of the true GT connectors of the 873 VAL frames that the pair mask of the pair head admits (`jends.py:214`: cos_turn >= COS_TURN, L >= MIN_CHORD, chord . ta > 0, chord . tb > 0), evaluated on the GT ends (ent: x, y, dx, dy of the entry; ext: of the exit; pairs: entry index, exit index).
A UNIT CHECK first: the same function on the keypoints of the stored dump of E324 s must give the stored `allowed` (with `valid`) for the first 60 frames of seed 0 (bit for bit).   usage (problem folder, CPU): wsl bash tools/env/_py14.sh mask_ceiling.py [--out temp/e364/mask_ceiling.json]"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import lane_gt_data as LD  # noqa: E402
import qendnet as QEN  # noqa: E402
import qends as QENDS  # noqa: E402
import jwire_data as JD  # noqa: E402

COS_TURN, MIN_CHORD = QEN.COS_TURN, QEN.MIN_CHORD


def parts(pa, ta, pb, tb):
    """-> dict of boolean arrays (n, m) for the four conditions of the mask: pa (n,2) entry positions, ta (n,2) entry headings, pb (m,2), tb (m,2) exit ones"""
    ch = pb[None, :, :] - pa[:, None, :]
    L = np.linalg.norm(ch, axis=-1)
    ta_ = np.broadcast_to(ta[:, None, :], ch.shape)
    tb_ = np.broadcast_to(tb[None, :, :], ch.shape)
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    return dict(turn=dot(ta_, tb_) >= COS_TURN, chord=L >= MIN_CHORD, fwd_a=dot(ch, ta_) > 0, fwd_b=dot(ch, tb_) > 0)


def unit_check(seed=0, n=60):
    ends = pickle.load(open(f"{ROOT}/runs/e324_s_s{seed}/ends_val.pkl", "rb"))[:n]
    bad = 0
    for e in ends:
        k = e["kp_pos"].shape[0] // 2
        p = parts(e["kp_pos"][:k].astype(float), e["kp_dir"][:k].astype(float), e["kp_pos"][k:].astype(float), e["kp_dir"][k:].astype(float))
        m = p["turn"] & p["chord"] & p["fwd_a"] & p["fwd_b"]
        v = np.asarray(e["valid"], bool)
        want = m & v[0][:, None] & v[1][None, :]
        got = np.asarray(e["allowed"], bool)
        bad += int((want != got).sum())
    return bad, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="temp/e364/mask_ceiling.json")
    a = ap.parse_args()
    bad, n = unit_check()
    print(f"unit check: the mask function on the keypoints of the first {n} stored frames differs from the stored `allowed` in {bad} cells")
    assert bad == 0, "the mask function is not the pair head's (heading convention?)"
    # `qends.end_table` builds its candidate list with `allowed` and returns None (an EMPTY table) when no pair of a frame passes it: the true pairs of such frames would be invisible. Patched to True while the table is built (the ends, the clusters and the true pairs do not depend on it).
    QENDS.allowed = lambda *a, **k: True
    LD.QData.DEFAULT_ENDS = True
    ds, dl = LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="exact", pixels=False, strict=True, ends=True)
    tot = dict(pairs=0, admitted=0, fail_turn=0, fail_chord=0, fail_fwd_a=0, fail_fwd_b=0, frames=0, frames_with_pairs=0)
    dumps = {s: {f["row"]: f for f in pickle.load(open(f"{ROOT}/runs/e324_s_s{s}/ends_val.pkl", "rb"))} for s in (0, 1)}
    real = {s: dict(pairs=0, both_ends_matched=0, reachable=0) for s in (0, 1)}
    for b in dl:
        for i in range(len(b["row"])):
            tot["frames"] += 1
            pr = b["pairs"][i].numpy()
            pr = pr[(pr[:, 0] >= 0) & (pr[:, 1] >= 0)]
            ent, ext = b["ent"][i].numpy().astype(float), b["ext"][i].numpy().astype(float)
            tot["frames_with_pairs"] += int(len(pr) > 0)
            for s in (0, 1):                                                                                  # the model's reality: a keypoint matched to each end of the true pair (4 m / 30 degrees, one to one: `jwire_data.match`) and the stored `allowed` True for that keypoint pair
                e = dumps[s][int(b["row"][i])]
                k = e["kp_pos"].shape[0] // 2
                v = np.asarray(e["valid"], bool)
                m_in = JD.match(e["kp_pos"][:k], e["kp_dir"][:k], v[0], ent, b["ent_m"][i].numpy())
                m_out = JD.match(e["kp_pos"][k:], e["kp_dir"][k:], v[1], ext, b["ext_m"][i].numpy())
                al = np.asarray(e["allowed"], bool)
                for (u, w) in pr:
                    real[s]["pairs"] += 1
                    iu, jw = np.where(m_in == u)[0], np.where(m_out == w)[0]
                    if len(iu) and len(jw):
                        real[s]["both_ends_matched"] += 1
                        real[s]["reachable"] += int(any(al[x, y] for x in iu for y in jw))
            for (u, w) in pr:
                p = parts(ent[u:u + 1, :2], ent[u:u + 1, 2:4], ext[w:w + 1, :2], ext[w:w + 1, 2:4])
                ok = all(bool(v[0, 0]) for v in p.values())
                tot["pairs"] += 1
                tot["admitted"] += int(ok)
                for k_ in ("turn", "chord", "fwd_a", "fwd_b"):
                    tot["fail_" + k_] += int(not p[k_][0, 0])
    share = tot["admitted"] / max(tot["pairs"], 1)
    res = dict(tot, model_reality={str(s): dict(real[s], reachable_of_matched=real[s]["reachable"] / max(real[s]["both_ends_matched"], 1), reachable_of_all=real[s]["reachable"] / max(real[s]["pairs"], 1)) for s in (0, 1)}, admitted_share=share, cos_turn=COS_TURN, min_chord=MIN_CHORD, unit_check_cells_differing=bad, reading=("FREE (>= 0.995)" if share >= 0.995 else f"COSTS {1 - share:.4f} of the true connectors"))
    json.dump(res, open(f"{ROOT}/{a.out}", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
