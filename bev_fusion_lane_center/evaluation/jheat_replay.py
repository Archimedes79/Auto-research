"""E299 step 0c (CPU, read-only): the heat-level replay of the E336 reading. CPU only, no training, no live code changed.

    wsl bash tools/env/_py14.sh evaluation/jheat_replay.py --seed 0 [--max 16]
    wsl bash tools/env/_py14.sh evaluation/jheat_replay.py selftest

The stored E312 model of the seed is run on the VAL frames (gt=None); its entry / exit heat (sigmoid of j1[:, 2:4]) and heading maps are kept. For every pair of consecutive keyframes of one
scene the heat of t-1 is warped into the lane frame of t (bilinear, zero outside the grid) and combined with the heat of t by the noisy-OR p_c = 1 - (1 - p_t)(1 - w p_w), w = 1.0 pre-registered
(0.5 / 0.25 reported in-sample). Keypoints from p_c exactly as JEnds.forward extracts them (3 x 3 maxima above PEAK_MIN, top K, soft-argmax over the 3 x 3 window of the combined map); the heading
is the heading map of frame t at the peak cell. Rule and quantities: experiments/E299_temporal-end-prior/experiment.md section 5b."""
import argparse
import json
import os
import pickle
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
import jends as JE  # noqa: E402
import jtime as JTM  # noqa: E402
import jwiring as JW  # noqa: E402

N1, HALF, RES, K, PEAK_MIN = JE.N1, JE.HALF, JE.RES, JE.K, JE.PEAK_MIN
WS = (1.0, 0.5, 0.25)           # 1.0 pre-registered, the others in-sample
WSTAR = 1.0


def extract(p, dirv, k=K):
    """p (2, N, N) heat probabilities, dirv (2, 2, N, N) unit heading maps -> pos (2K, 2), dir (2K, 2), valid (2K,) as JEnds.forward (entries first, then exits)"""
    c = JE._centres("cpu")
    off = torch.tensor([-1.0, 0.0, 1.0])
    dx, dy = torch.meshgrid(off, off, indexing="ij")
    dx, dy = dx.flatten(), dy.flatten()
    pos, D, valid = [], [], []
    for kk in (0, 1):
        pk = p[kk]
        mx = F.max_pool2d(pk[None, None], 3, 1, 1)[0, 0]
        score = torch.where((pk >= mx) & (pk > PEAK_MIN), pk, torch.zeros_like(pk)).flatten()
        v, ix = score.topk(k)
        unf = F.unfold(pk[None, None], 3, padding=1)[0]
        w = unf[:, ix]
        wn = w / w.sum(0, keepdim=True).clamp_min(1e-6)
        i0, j0 = ix // N1, ix % N1
        x = c[i0] + (wn * dx[:, None] * RES).sum(0)
        y = c[j0] + (wn * dy[:, None] * RES).sum(0)
        pos.append(torch.stack([x, y], -1))
        D.append(dirv[kk].flatten(1)[:, ix].T)
        valid.append(v > 0)
    return torch.cat(pos).numpy(), torch.cat(D).numpy(), torch.cat(valid).numpy()


def warp_heat(p_prev, f0, f1):
    """(2, N, N) heat of the frame with lane frame f0 in the lane frame f1; target cell centres x_i = c[i] (dim 1), y_j = c[j] (dim 2); inverse of jtime.warp: p0 = (p1 A1^T + c1 - c0) A0"""
    (A0, c0), (A1, c1) = f0, f1
    c = JE._centres("cpu").numpy()
    X, Y = np.meshgrid(c, c, indexing="ij")
    P1 = np.stack([X, Y], -1).reshape(-1, 2)
    P0 = (P1 @ A1.T + c1 - c0) @ A0
    g = np.stack([P0[:, 1], P0[:, 0]], -1).reshape(N1, N1, 2) / HALF              # grid_sample: [..., 0] = column (dim 2 = y), [..., 1] = row (dim 1 = x)
    out = F.grid_sample(p_prev[None], torch.tensor(g, dtype=torch.float32)[None], mode="bilinear", padding_mode="zeros", align_corners=False)
    return out[0]


def combine(pt, pw, w):
    return 1.0 - (1.0 - pt) * (1.0 - w * pw)


def _sets(m):
    return {int(x) for x in m if x >= 0}


def pair_stats(maps0, maps1, f0, f1, g1, ws=WS, rng=None):
    """maps = (p (2, N, N), dirv (2, 2, N, N)) of t-1 and t; -> {arm: counts} over both kinds. Arms: 'w<w>' the noisy-OR with the warped previous heat; 'N1' the heat of t with ITSELF (no time information);
    'N2' the previous heat NOT warped. Also the heat-visible ceiling (section 5b amendment): of the GT ends the baseline misses, those with a warped-previous-heat value >= 0.05 within 4 m, against random places."""
    p0, _ = maps0
    p1, d1 = maps1
    pw = warp_heat(p0, f0, f1)
    pos_b, dir_b, val_b = extract(p1, d1)
    k = len(pos_b) // 2
    arms = {f"w{w}": combine(p1, pw, w) for w in ws}
    arms["N1"] = combine(p1, p1, 1.0)
    arms["N2"] = combine(p1, p0, 1.0)
    res = {}
    for name, pc in arms.items():
        pos_c, dir_c, val_c = extract(pc, d1)
        r = dict(gt=0, base_tp=0, comb_tp=0, gained=0, lost=0, base_delivered=int(val_b.sum()), comb_delivered=int(val_c.sum()), new=0, true_new=0, false_new=0, false_new_near=0, false_new_isolated=0,
                 vanished=0, vanished_matched=0)
        for kind, (G, m) in enumerate(((g1["ent"], g1["ent_m"]), (g1["ext"], g1["ext_m"]))):
            sl = slice(kind * k, (kind + 1) * k)
            mb = JW.match(pos_b[sl], dir_b[sl], val_b[sl], G, m)
            mc = JW.match(pos_c[sl], dir_c[sl], val_c[sl], G, m)
            sb, sc = _sets(mb), _sets(mc)
            missed = {int(x) for x in np.nonzero(m)[0]} - sb
            r["gt"] += int(m.sum())
            r["base_tp"] += len(sb)
            r["comb_tp"] += len(sc)
            r["gained"] += len(sc - sb)
            r["lost"] += len(sb - sc)
            Eb = np.concatenate([pos_b[sl], dir_b[sl]], 1)
            Ec = np.concatenate([pos_c[sl], dir_c[sl]], 1)
            cp = JTM.counterpart(Ec, Eb, val_c[sl], val_b[sl])                       # comb as "previous", base as "current": cp < 0 = a keypoint of p_c without a baseline counterpart
            vb_pos = pos_b[sl][val_b[sl]]
            for i in np.nonzero(val_c[sl])[0]:
                if cp[i] < 0:
                    r["new"] += 1
                    if mc[i] >= 0 and int(mc[i]) in missed:
                        r["true_new"] += 1
                    else:
                        r["false_new"] += 1
                        near = len(vb_pos) and float(np.linalg.norm(vb_pos - pos_c[sl][i], axis=1).min()) <= 4.0
                        r["false_new_near" if near else "false_new_isolated"] += 1
            cpb = JTM.counterpart(Eb, Ec, val_b[sl], val_c[sl])                      # baseline keypoints without a counterpart in p_c
            for i in np.nonzero(val_b[sl])[0]:
                if cpb[i] < 0:
                    r["vanished"] += 1
                    r["vanished_matched"] += int(mb[i] >= 0)
        assert r["comb_tp"] - r["base_tp"] == r["gained"] - r["lost"], r
        res[name] = r
    # heat-visible ceiling: baseline-missed GT ends with warped previous heat >= 0.05 within 4 m (any heading) vs the same number of random places in the same frame
    c = JE._centres("cpu").numpy()
    X, Y = np.meshgrid(c, c, indexing="ij")
    ceil = dict(missed=0, visible=0, random_visible=0)
    for kind, (G, m) in enumerate(((g1["ent"], g1["ent_m"]), (g1["ext"], g1["ext_m"]))):
        sl = slice(kind * k, (kind + 1) * k)
        mb = JW.match(pos_b[sl], dir_b[sl], val_b[sl], G, m)
        missed = sorted({int(x) for x in np.nonzero(m)[0]} - _sets(mb))
        h = pw[kind].numpy()

        def vis(xy):
            return bool(h[((X - xy[0]) ** 2 + (Y - xy[1]) ** 2) <= 16.0].max(initial=0.0) >= 0.05)
        for gi in missed:
            ceil["missed"] += 1
            ceil["visible"] += int(vis(G[gi, :2]))
            ceil["random_visible"] += int(vis(rng.uniform(-HALF + 1, HALF - 1, 2)))
    res["ceiling"] = ceil
    return res


def selftest():
    import math
    # warp: identity pose keeps the heat; a pose that moves the ego 5 m forward moves a blob from x = 10 to x = 5 (jtime.warp semantics)
    p = torch.zeros(2, N1, N1)
    p[0, 60, 50] = 1.0                                                              # cell (i = 60, j = 50): x = 10.5, y = 0.5
    I = (np.eye(2), np.zeros(2))
    assert torch.allclose(warp_heat(p, I, I), p, atol=1e-3)
    f1 = (np.eye(2), np.array([5.0, 0.0]))
    q = warp_heat(p, I, f1)
    i, j = np.unravel_index(int(q[0].argmax()), q[0].shape)
    assert (i, j) == (55, 50), (i, j)                                                # x 10.5 -> 5.5: the cell moved with the ego
    E = np.array([[10.5, 0.5, 1.0, 0.0]])
    w = JTM.warp(E, I, f1)
    assert abs(w[0, 0] - 5.5) < 1e-9 and abs(w[0, 1] - 0.5) < 1e-9                  # the same convention as the point warp
    # a rotation by 90 degrees: the heat and the point warp agree on where the blob lands
    th = math.pi / 2
    A = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    f2 = (A, np.array([1.0, -2.0]))
    q = warp_heat(p, I, f2)
    wp = JTM.warp(E, I, f2)[0, :2]
    c = JE._centres("cpu").numpy()
    i, j = np.unravel_index(int(q[0].argmax()), q[0].shape)
    assert abs(c[i] - wp[0]) <= 1.0 and abs(c[j] - wp[1]) <= 1.0, ((c[i], c[j]), wp)
    # noisy-OR: two looks of 0.07 give 0.1351 > PEAK_MIN, one look does not; w = 0 is the baseline
    a = torch.full((2, 3, 3), 0.07)
    assert float(combine(a, a, 1.0)[0, 0, 0]) > PEAK_MIN > float(a[0, 0, 0])
    assert torch.allclose(combine(a, a, 0.0), a, atol=1e-6)
    # extraction: one clear peak -> one valid keypoint at the cell centre, heading from the map
    pk = torch.zeros(2, N1, N1)
    pk[0, 30, 40] = 0.8
    dv = torch.zeros(2, 2, N1, N1)
    dv[..., 0, :, :] = 1.0
    pos, D, val = extract(pk, dv)
    assert val.sum() == 1 and abs(pos[0, 0] - (c[30])) < 1e-4 and abs(pos[0, 1] - (c[40])) < 1e-4 and D[0, 0] == 1.0
    # the statistics: a GT end the baseline misses and the combined heat finds counts as gained / true_new; a baseline end that disappears counts as lost
    G = np.array([[c[30], c[40], 1.0, 0.0], [c[10], c[10], 1.0, 0.0]])
    g1 = dict(ent=G, ent_m=np.array([True, True]), ext=np.zeros((2, 4)), ext_m=np.array([False, False]))
    p1 = torch.zeros(2, N1, N1)
    p1[0, 10, 10] = 0.5                                                              # the baseline sees the second end only
    p0 = torch.zeros(2, N1, N1)
    p0[0, 30, 40] = 0.09                                                             # a sub-threshold look at the first end one frame earlier, ego did not move
    p1[0, 30, 40] = 0.05                                                             # and a sub-threshold look now
    d1 = torch.zeros(2, 2, N1, N1)
    d1[:, 0] = 1.0
    r = pair_stats((p0, d1), (p1, d1), I, I, g1, ws=(1.0, 0.0), rng=np.random.RandomState(0))
    assert r["w1.0"]["base_tp"] == 1 and r["w1.0"]["comb_tp"] == 2 and r["w1.0"]["gained"] == 1 and r["w1.0"]["lost"] == 0 and r["w1.0"]["true_new"] == 1 and r["w1.0"]["false_new"] == 0, r["w1.0"]
    assert r["N2"]["gained"] == 1, r["N2"]                                           # identity pose here: the unwarped heat equals the warped one
    assert r["ceiling"]["missed"] == 1 and r["ceiling"]["visible"] == 1, r["ceiling"]
    assert r["w0.0"]["gained"] == 0 and r["w0.0"]["new"] == 0, r["w0.0"]
    assert r["N1"]["gained"] == 0, r["N1"]                                          # 0.05 with itself is 0.0975 < 0.1: the self-OR alone does not find the end
    print("heat_replay selftest ok")


def run(a):
    import lane_gt_data as LD
    import jends_eval as JEV
    run_name = f"e312_p_s{a.seed}"
    args = argparse.Namespace(seed=a.seed, run=run_name)
    _, ends, gt, frames, scenes, infos = JTM._setup(args)
    pairs = JTM._pairs_of_frames(scenes, ends)
    if a.max:
        keep = set()
        for t0, t1 in pairs[: a.max]:
            keep |= {t0, t1}
    else:
        keep = None
    m, _ = JEV.load_model(run_name, "cpu")
    LD.QData.DEFAULT_JUNCTION = True
    ds, dl = LD.loader("val_clean", bs=8, workers=0, shuffle=False, map_mode="error", pixels=False)
    maps, sane = {}, dict(frames=0, same_count=0, max_dpos=0.0)
    with torch.no_grad():
        for b in dl:
            toks = list(b["token"])
            if keep is not None and not any(t in keep for t in toks):
                continue
            _, out = m.jmodel(b["x"].float(), joint=False)
            j1 = out["j1"].float()
            p = j1[:, 2:4].sigmoid()
            dirv = torch.stack([JE._unit(j1[:, 4:6]), JE._unit(j1[:, 6:8])], 1)
            for i, t in enumerate(toks):
                maps[t] = (p[i].clone(), dirv[i].clone())
                if t in ends:                                                            # the sanity check against the stored dump of the same model
                    pos, D, val = extract(p[i], dirv[i])
                    e = ends[t]
                    sane["frames"] += 1
                    sane["same_count"] += int((np.asarray(e["valid"]).reshape(-1) == val).all())
                    dv_ = np.asarray(e["valid"]).reshape(-1)
                    A_, B_ = pos[val], np.asarray(e["kp_pos"])[dv_]
                    same_set = False
                    if len(A_) and len(B_):                                            # as SETS: saturated peaks tie in the top-K, so the ORDER and, at the K-cut, the members may differ between CPU and GPU
                        dd = np.linalg.norm(A_[:, None] - B_[None], axis=-1)
                        same_set = len(A_) == len(B_) and dd.min(1).max() <= 0.05 and dd.min(0).max() <= 0.05
                        if same_set:
                            sane["max_dpos"] = max(sane["max_dpos"], float(dd.min(1).max()))
                    sane["same_set"] = sane.get("same_set", 0) + int(same_set)
                    if not same_set:                                                   # a differing frame must have a tie at the K-cut of some kind: more than K peaks and the K-th and K+1-th score equal
                        tie = False
                        for kk in (0, 1):
                            pk = p[i][kk]
                            mx = F.max_pool2d(pk[None, None], 3, 1, 1)[0, 0]
                            sc = torch.where((pk >= mx) & (pk > PEAK_MIN), pk, torch.zeros_like(pk)).flatten().sort(descending=True).values
                            tie = tie or bool((sc[K] > 0) and abs(float(sc[K - 1] - sc[K])) < 1e-6)
                            cand = ((pk >= mx) & (pk > PEAK_MIN)).float()               # a plateau: equal-valued saturated cells next to each other are ALL peaks, and which of them the top-K takes is a tie
                            plateau = bool(((F.avg_pool2d(cand[None, None], 3, 1, 1)[0, 0] * 9 >= 1.5) & (cand > 0)).any())
                            tie = tie or plateau
                        sane.setdefault("differing_frames", []).append((t, tie))
    sane["unequal_frames"] = sane["frames"] - sane["same_set"]
    sane["differing_without_cut_tie"] = sum(1 for _, tie in sane.get("differing_frames", []) if not tie)
    sane["differing_frames"] = len(sane.get("differing_frames", []))
    print("baseline sanity against ends_val.pkl:", sane, flush=True)
    assert sane["frames"] and sane["same_set"] >= 0.90 * sane["frames"] and sane["max_dpos"] <= 0.05, "the baseline extraction does not reproduce the stored keypoints: stop"
    rng = np.random.RandomState(1234)
    tot, ceil = {}, dict(missed=0, visible=0, random_visible=0)
    npairs = 0
    for t0, t1 in pairs:
        if t0 not in maps or t1 not in maps:
            continue
        r = pair_stats(maps[t0], maps[t1], JTM.lane_frame(infos[t0]), JTM.lane_frame(infos[t1]), gt[frames[t1].row], rng=rng)
        npairs += 1
        for name, v in r.items():
            if name == "ceiling":
                for kk in ceil:
                    ceil[kk] += v[kk]
            elif name not in tot:
                tot[name] = dict(v)
            else:
                for kk in tot[name]:
                    tot[name][kk] += v[kk]
    out = dict(seed=a.seed, pairs=npairs, sanity=sane, w_star=WSTAR, arms={}, heat_visible_ceiling=dict(ceil, visible_share=ceil["visible"] / max(ceil["missed"], 1), random_share=ceil["random_visible"] / max(ceil["missed"], 1)))
    for name, t in tot.items():
        d = dict(t)
        d["net_gain_share"] = (t["comb_tp"] - t["base_tp"]) / t["gt"]
        d["gross_gain_share"] = t["gained"] / t["gt"]
        d["base_recall"], d["comb_recall"] = t["base_tp"] / t["gt"], t["comb_tp"] / t["gt"]
        d["base_precision"] = t["base_tp"] / max(t["base_delivered"], 1)
        d["comb_precision"] = t["comb_tp"] / max(t["comb_delivered"], 1)
        out["arms"][name] = d
    star = out["arms"][f"w{WSTAR}"]
    n1, n2 = out["arms"]["N1"]["net_gain_share"], out["arms"]["N2"]["net_gain_share"]
    out["rule_w_star"] = dict(
        net_gain_share=star["net_gain_share"], true_new=star["true_new"], false_new=star["false_new"],
        a_net_gain_share_ge_0_05=bool(star["net_gain_share"] >= 0.05), at_the_bar_0_045_0_055=bool(0.045 <= star["net_gain_share"] <= 0.055),
        temporal_excess_over_nulls=star["net_gain_share"] - max(n1, n2), a_temporal=bool(star["net_gain_share"] >= 0.05 and not (0.045 <= star["net_gain_share"] <= 0.055) and star["net_gain_share"] - max(n1, n2) >= 0.02),
        b_true_new_ge_false_new=bool(star["true_new"] >= star["false_new"]))
    print(json.dumps(out, indent=1))
    if not a.max:
        # step 0b numbers of the baseline must be reproduced (section 5b amendment): recall 0.787 / 0.790, precision 0.620 / 0.602 within 0.003
        want = {0: (0.787, 0.620), 1: (0.790, 0.602)}[a.seed]
        got = (star["base_recall"], star["base_precision"])
        out["baseline_matches_step_0b"] = bool(abs(got[0] - want[0]) <= 0.003 and abs(got[1] - want[1]) <= 0.003)
        os.makedirs(os.path.join(ROOT, "temp", "jtime"), exist_ok=True)
        json.dump(out, open(os.path.join(ROOT, "temp", "jtime", f"heat_replay_s{a.seed}.json"), "w"), indent=1)
        print("baseline matches step 0b:", out["baseline_matches_step_0b"], got, want)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else run(a)
