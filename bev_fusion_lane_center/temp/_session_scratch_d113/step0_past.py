import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_seq_data as LS
import lane_tok_data as LT
import lane_tokcache as LC
import p16_data as PD

seed = sys.argv[1] if len(sys.argv) > 1 else "0"
split = sys.argv[2] if len(sys.argv) > 2 else "val"
c = LC.TokCache(os.path.join(R, f"runs/e365_cache/be292_a_{seed}_K512_9ad1d4ce30_v1/{split}"))
pose = dict(np.load(f"{PD.D}/poses_{split}.npz"))
row2i = {int(r): i for i, r in enumerate(c.rows)}
ds = LT.TokData(split, res=2.0, feats=False)
n = len(c) if split == "val" else 1500
idx = np.linspace(0, len(c) - 1, n).astype(int) if split != "val" else np.arange(len(c))
tot = dict(gt=0, found=0, missed=0, missed_weak=0, missed_past=0, missed_weak_past=0, missed_absent=0, frames=0, nopast=0)
wk = []
cnt = dict(tw_s=0, tw=0, fw_s=0, fw=0, td_s=0, td=0, fd_s=0, fd=0)
RAD, COS = 1.5, 0.5


def near_any(pos, dirv, valid, gpos, gdirv):
    if len(gpos) == 0:
        return np.zeros(len(pos), bool)
    d = np.linalg.norm(pos[:, None] - gpos[None], axis=2)
    cs = dirv @ gdirv.T
    return (valid[:, None] & (d <= RAD) & (cs >= COS)).any(1)


for i in idx:
    pr = int(pose["prev"][int(c.rows[i])])
    t = ds.table(i)
    pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
    deliv = valid & (ep >= 0.5)
    gpos, gdir = t["pos"], t["dirv"]
    tot["frames"] += 1
    if pr < 0 or pr not in row2i:
        tot["nopast"] += 1
        continue
    ip = row2i[pr]
    Rm, tv = LS.rel_pose(pose["R"][int(c.rows[i])], pose["t"][int(c.rows[i])], pose["R"][pr], pose["t"][pr])
    ppos = np.array(c.f["pos"][ip], float) @ Rm.T.astype(float) + tv
    pdir = np.array(c.f["dirv"][ip], float) @ Rm.T.astype(float)
    pdel = np.array(c.f["valid"][ip]) & (np.array(c.f["e_p"][ip], float) >= 0.5)
    pep = np.array(c.f["e_p"][ip], float)
    # GT nodes of t: found by a delivered token / present as a weak token / absent; recoverable from a delivered past token
    M = len(gpos)
    if M:
        dd = np.linalg.norm(deliv_pos := pos[deliv][:, None] - gpos[None], axis=2) if deliv.any() else np.full((1, M), 9.9)
        cs = (dirv[deliv] @ gdir.T) if deliv.any() else np.full((1, M), -1.0)
        found = ((dd <= RAD) & (cs >= COS)).any(0)
        weak = valid & ~deliv
        dw = np.linalg.norm(pos[weak][:, None] - gpos[None], axis=2) if weak.any() else np.full((1, M), 9.9)
        cw = (dirv[weak] @ gdir.T) if weak.any() else np.full((1, M), -1.0)
        has_weak = ((dw <= RAD) & (cw >= COS)).any(0)
        dp = np.linalg.norm(ppos[pdel][:, None] - gpos[None], axis=2) if pdel.any() else np.full((1, M), 9.9)
        cp = (pdir[pdel] @ gdir.T) if pdel.any() else np.full((1, M), -1.0)
        past = ((dp <= RAD) & (cp >= COS)).any(0)
        tot["gt"] += M
        tot["found"] += int(found.sum())
        miss = ~found
        tot["missed"] += int(miss.sum())
        tot["missed_weak"] += int((miss & has_weak).sum())
        tot["missed_absent"] += int((miss & ~has_weak).sum())
        tot["missed_past"] += int((miss & past).sum())
        tot["missed_weak_past"] += int((miss & has_weak & past).sum())
    # weak tokens: true (near a GT node) or false; supported by a delivered warped token of t-1 within 1.5 m, cos >= 0.5
    for name, sel in (("w", valid & ~deliv), ("d", deliv)):
        if not sel.any():
            continue
        tr = near_any(pos[sel], dirv[sel], np.ones(sel.sum(), bool), gpos, gdir)
        d = np.linalg.norm(pos[sel][:, None] - ppos[pdel][None], axis=2) if pdel.any() else np.full((sel.sum(), 1), 9.9)
        cs = dirv[sel] @ pdir[pdel].T if pdel.any() else np.full((sel.sum(), 1), -1.0)
        sup = ((d <= RAD) & (cs >= COS)).any(1)
        k = "w" if name == "w" else "d"
        if name == "w":
            wk.append(np.stack([ep[sel], tr.astype(float), sup.astype(float)], 1))
        for lab, key in ((True, "t"), (False, "f")):
            m = tr == lab
            cnt[f"{key}{k}_s"] += int((sup & m).sum())
            cnt[f"{key}{k}"] += int(m.sum())
print("split", split, "seed", seed, "frames", tot["frames"], "without past", tot["nopast"])
g = tot["gt"]
print("GT nodes of frames with a past: %d; found by a delivered token (e >= 0.5) %.4f; missed %.4f" % (g, tot["found"] / g, tot["missed"] / g))
print("  of the MISSED: a weak token (0.05 < p < 0.5) is near %.4f, no token at all %.4f; a delivered token of t-1 (warped) is near %.4f; weak AND past support %.4f" % (
    tot["missed_weak"] / tot["missed"], tot["missed_absent"] / tot["missed"], tot["missed_past"] / tot["missed"], tot["missed_weak_past"] / tot["missed"]))
for k, nm in (("w", "weak tokens (valid, p < 0.5)"), ("d", "delivered tokens (p >= 0.5)")):
    print("%s: true (near a GT node): %d, supported by the past %.4f | false: %d, supported %.4f" % (nm, cnt[f"t{k}"], cnt[f"t{k}_s"] / max(cnt[f"t{k}"], 1), cnt[f"f{k}"], cnt[f"f{k}_s"] / max(cnt[f"f{k}"], 1)))

W = np.concatenate(wk)
print("weak tokens by existence bin: p range | tokens | true share | supported (true) | supported (false)")
for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
    m = (W[:, 0] >= lo) & (W[:, 0] < hi)
    t_ = m & (W[:, 1] > 0.5)
    f_ = m & (W[:, 1] < 0.5)
    print("  %.2f - %.2f | %7d | %.3f | %.3f | %.3f" % (lo, hi, m.sum(), t_.sum() / max(m.sum(), 1), W[t_, 2].mean(), W[f_, 2].mean()))
