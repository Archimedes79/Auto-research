"""G0 of E271: the ground truth through the chain must reproduce itself (real VAL frames, CPU).

    wsl bash tools/env/_py14.sh training/grid/test_oracle.py [--n 100]

(a) stages 1 + 2: the GT lanes become the head's output (heat from the exact distance, nearest heading, offset, and a keep gate = 1 on the GT vertices: the
    Douglas-Peucker vertices of each GT polyline, plus a vertex wherever a segment is longer than `MAXSEG` m; a named off-path ORACLE step), go through
    `soft_lines.sparsify` and the read-out of `evaluation/vertex_eval.py`, and are scored with `lane_task.measures.vertex_*`:
    prec1 >= 0.98, cov5 >= 0.95, vgt100 <= 14.3. Degraded copies must fall: shifted 1 m across, 30 % of the vertices dropped, headings flipped, a dense gate.
(b) stage 4: the oracle vertices through the START-value `Assigner` with the exact SD map: the share of vertices whose most probable road equals the road a
    geometric rule gives (nearest road token within 3 m with a heading within 30 degrees) >= 0.97. The map has no explicit links and holds the GT lanes' own
    carriageways, so this tests the frame conventions and the direction handling on real data, not the learning.
Failing either part means the representation is the limit: no training.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/grid", "evaluation", "training/data"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import assign as AS  # noqa: E402
import soft_lines as SL  # noqa: E402
import targets as TG  # noqa: E402
import vertex_eval as VE  # noqa: E402
from lane_task import measures as M  # noqa: E402

MAXSEG = 9.0            # a vertex at least every 9 m: with a 5 m reach on both sides the lane is covered (half the spacing = 4.5 m < 5 m)
DP_EPS = 0.2            # the tolerance of the prototype's output stage (dp_eps)


def gt_vertices(g, eps=DP_EPS, maxseg=MAXSEG):
    """oracle gate: Douglas-Peucker vertices of every GT polyline, long segments subdivided -> positions (N, 2), tangents (N, 2) (metric frame x right, y forward)"""
    import sd_lib
    V, T = [], []
    for p in g["pts"]:
        p = np.asarray(p, np.float64)
        if len(p) < 2:
            continue
        idx = sd_lib._dp(p, eps)
        q = p[idx]
        pts = [q[0]]
        for a, b in zip(q[:-1], q[1:]):
            n = int(np.ceil(np.linalg.norm(b - a) / maxseg))
            pts += [a + (b - a) * k / n for k in range(1, n + 1)]
        pts = np.asarray(pts)
        t = np.gradient(pts, axis=0)
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        V.append(pts), T.append(t)
    return (np.concatenate(V), np.concatenate(T)) if V else (np.zeros((0, 2)), np.zeros((0, 2)))


def oracle_output(g, gate_pos=None, gate_tan=None, gate_all=False):
    """the head output (1, 6, 100, 100) = [p logit, cos, sin, ox, oy, s logit] that a perfect head would give for the GT graph g"""
    G, T = VE.gt_dense(g)
    G = np.concatenate([G, np.full((8, 2), 1e4)]).astype(np.float32)
    T = np.concatenate([T, np.zeros((8, 2))]).astype(np.float32)
    d, off, head, _ = TG.cell_targets(torch.from_numpy(G)[None], torch.from_numpy(T)[None])
    p = TG.heat(d).clamp(1e-4, 1 - 1e-4)
    out = torch.zeros(1, 6, 100, 100)
    out[:, 0] = torch.log(p / (1 - p))
    out[:, 1:3] = head
    out[:, 3:5] = torch.atanh(off.clamp(-0.5, 0.5) * 2 * 0.999)             # sparsify: offset = 0.5 tanh(x)
    out[:, 5] = 6.0 if gate_all else -6.0
    if gate_pos is not None and len(gate_pos):
        ix = np.floor((gate_pos[:, 0] + 50.0) / 1.0).astype(int)
        iy = np.floor((gate_pos[:, 1] + 50.0) / 1.0).astype(int)
        ok = (ix >= 0) & (ix < 100) & (iy >= 0) & (iy < 100)
        ix, iy, pos, tan = ix[ok], iy[ok], gate_pos[ok], gate_tan[ok]
        out[0, 5, ix, iy] = 6.0
        out[0, 1, ix, iy] = torch.from_numpy(tan[:, 0]).float()
        out[0, 2, ix, iy] = torch.from_numpy(tan[:, 1]).float()
        o = np.stack([(pos[:, 0] + 50.0) - (ix + 0.5), (pos[:, 1] + 50.0) - (iy + 0.5)], 1).clip(-0.5, 0.5)       # the vertex's own position in the cell, in cells
        out[0, 3, ix, iy] = torch.atanh(torch.from_numpy(o[:, 0]).float() * 2 * 0.999)
        out[0, 4, ix, iy] = torch.atanh(torch.from_numpy(o[:, 1]).float() * 2 * 0.999)
    return out


def vertices_of(out, thr=0.3):
    sp = SL.sparsify(out)[0]
    V, H, w = VE.read_out(sp.numpy(), thr)
    return V, H


def part_a(gts, report=print):
    rows = {}
    for name in ("oracle", "shift 1 m", "drop 30 %", "flipped heading", "dense vertices"):
        cs = []
        for i, g in enumerate(gts):
            gp, gt_ = gt_vertices(g)
            out = oracle_output(g, gp, gt_)
            V, H = vertices_of(out)
            rng = np.random.RandomState(i)
            if name == "shift 1 m" and len(V):
                t = np.stack([-H[:, 1], H[:, 0]], 1)
                V = V + 1.0 * t                                      # a metre across each vertex's heading
            if name == "drop 30 %" and len(V):
                k = rng.rand(len(V)) > 0.3
                V, H = V[k], H[k]
            if name == "flipped heading":
                H = -H
            if name == "dense vertices":                              # not through the read-out (its 3 x 3 maximum caps the density): a vertex every 1 m of the GT
                V, H = VE.gt_dense(g)[0][::2], VE.gt_dense(g)[1][::2]
            cs.append(VE.frame_counts(V, H, g))
        rows[name] = M.vertex_numbers(VE.total(cs))
        report(f"  {name:16s}", {k: round(float(v), 3) for k, v in rows[name].items()})
    o = rows["oracle"]
    assert o["prec1"] >= 0.98 and o["cov5"] >= 0.95 and o["vgt100"] <= 14.3, ("G0 (a) failed", o)
    assert rows["shift 1 m"]["prec1"] < 0.7 and rows["drop 30 %"]["cov5"] < o["cov5"] - 0.05 and rows["flipped heading"]["prec1"] < 0.05 and rows["dense vertices"]["vgt100"] > 50 and rows["dense vertices"]["cov5"] > 0.99, rows
    return rows


def part_b(items, report=print):
    """items: [(gt graph, sd map)] -> share of oracle vertices whose most probable road is a PLAUSIBLE road: its nearest token lies within 0.5 m of the nearest
    token overall and within 30 degrees of the heading. Strict agreement with the single nearest road is reported too: the SD map holds duplicate roads (two
    carriageways within 0.5 m of each other, 14 % of the vertices), where the nearest one is a coin toss and a strict `road_acc` would measure the coin."""
    from scipy.spatial import cKDTree
    mod = AS.Assigner()
    strict = plaus = lab = dust = tot = 0
    for g, sd in items:
        gp, gt_ = gt_vertices(g)
        sd = dict(roads=sd["roads"], route=[])                      # the route copies are a second token set on the same roads: not used here
        mp = AS.map_tokens(sd)
        if len(gp) == 0 or len(mp["pos"]) == 0:
            continue
        pos = np.stack([gp[:, 1], -gp[:, 0]], 1).astype(np.float32)   # metric frame (x right, y forward) -> grid frame (x forward, y left)
        hdg = np.stack([gt_[:, 1], -gt_[:, 0]], 1).astype(np.float32)
        d, j = cKDTree(mp["pos"]).query(pos)
        ok = (d < 3.0) & ((hdg * mp["dir"][j]).sum(1) > np.cos(np.deg2rad(30)))
        with torch.no_grad():
            out = mod(torch.from_numpy(pos), torch.from_numpy(hdg), torch.ones(len(pos)), AS.to_torch(mp))
        P = out["P_road"].numpy()
        R = mp["n_ids"]
        best = P[:, :R].argmax(1)
        for i in np.nonzero(ok)[0]:
            m = mp["rid"] == best[i]
            dd = np.linalg.norm(mp["pos"][m] - pos[i], axis=1)
            jj = int(dd.argmin())
            plaus += int(dd[jj] <= d[i] + 0.5 and float(hdg[i] @ mp["dir"][m][jj]) > np.cos(np.deg2rad(30)))
        strict += int((best[ok] == mp["rid"][j][ok]).sum())
        lab += int(ok.sum())
        dust += int((P[ok, R] > 0.5).sum())
        tot += len(pos)
    acc = plaus / max(lab, 1)
    report(f"  roads: plausible {acc:.3f}, strict {strict / max(lab, 1):.3f} of {lab} labelled vertices ({tot} in all), {dust} of the labelled went to the dustbin")
    assert acc >= 0.97 and lab > 0.5 * tot, ("G0 (b) failed", acc, lab, tot)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    import p16_data as PD
    import p8_lib as L
    ds = PD.BuilderData("val_clean", train=False)
    ks = np.linspace(0, len(ds.rows) - 1, a.n).astype(int)
    gts = [ds.gt[int(ds.rows[k])] for k in ks]
    sds = L.cache("sd", ds.part)
    print(f"G0 (a): GT -> sparsify -> read-out -> vertex measures on {len(ks)} VAL frames")
    part_a(gts)
    print("G0 (b): oracle vertices -> START-value Assigner with the exact SD map")
    items = [(g, sds[int(ds.rows[k])]) for g, k in zip(gts, ks) if sds[int(ds.rows[k])] is not None]
    part_b(items)
    print("G0 ok: the representation reproduces the ground truth")


if __name__ == "__main__":
    main()
