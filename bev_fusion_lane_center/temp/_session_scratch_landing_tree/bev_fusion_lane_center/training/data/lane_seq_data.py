"""E354: sequences of K consecutive keyframes of ONE scene around `lane_gt_data.QData`, with the relative ego poses (the data of the recursive feature state, `grid/jrecur.py`).

    ds, dl = loader("train", K=3, bs=4, workers=4)        # one item = K frames of QData (time order) + the K - 1 relative poses
    batch             ONE batch of `lane_gt_data.collate` over the B * K frames (what the E341 loss reads), frame k of sequence b at index k * B + b: [k * B:(k + 1) * B] = the frames of time step k
    batch["K"]        K; batch["rel_R"] (B, K - 1, 2, 2), batch["rel_t"] (B, K - 1, 2): [:, k - 1] = the lane frame of frame k - 1 in that of frame k (`rel_pose`)
    order = stream_order(pose, rows)                      # the read-out order: [(row, reset)] by scene and time, reset = the row has no previous keyframe in the list

Poses: `~/exp_gcar_b123/poses_{train,val}.npz` (p_global = t + R p_lane, the lane frame x right / y forward; `prev` = the row of the previous keyframe of the scene or -1; `ts`, `scene`). A sequence is K rows linked by `prev` that ALL exist
in the QData (a row without stored lane GT is not in it): no sequence crosses a scene start or a missing row. Tests: grid/test_jrecur.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lane_gt_data as LD  # noqa: E402
import p16_data as PD  # noqa: E402


def rel_pose(Rc, tc, Rp, tp):
    """the lane frame of the PREVIOUS keyframe in that of the current one: p_cur = R p_prev + t with R = Rc^T Rp, t = Rc^T (tp - tc); the inputs are rows of `poses_<part>.npz`"""
    Rc, tc, Rp, tp = (np.asarray(a, np.float64) for a in (Rc, tc, Rp, tp))
    return (Rc.T @ Rp).astype(np.float32), (Rc.T @ (tp - tc)).astype(np.float32)


def chains(prev, rows, K):
    """every run of K rows in time order, each the `prev` of the next, all of them in `rows` -> list of K-tuples of rows"""
    have, out = set(int(r) for r in rows), []
    for r in sorted(have):
        seq = [r]
        while len(seq) < K and int(prev[seq[0]]) in have:                    # (-1 = a scene start is never a row)
            seq.insert(0, int(prev[seq[0]]))
        if len(seq) == K:
            out.append(tuple(seq))
    return out


def stream_order(pose, rows):
    """the rows by scene, then time -> [(row, reset)]; reset = the previous keyframe of the row (`pose['prev']`) is not the row before it in the list (a scene start or a gap): the state starts empty there"""
    rows = sorted((int(r) for r in rows), key=lambda r: (int(pose["scene"][r]), float(pose["ts"][r])))
    return [(r, i == 0 or int(pose["prev"][r]) != rows[i - 1]) for i, r in enumerate(rows)]


class SeqData(torch.utils.data.Dataset):
    def __init__(self, split="train", K=3, **kw):
        self.ds, self.K = LD.QData(split, **kw), K
        self.pose = dict(np.load(f"{PD.D}/poses_{self.ds.bd.part}.npz"))
        self.i_of_row = {int(self.ds.bd.rows[k]): i for i, k in enumerate(self.ds.ks)}                   # row of the part -> index of the QData
        self.seqs = chains(self.pose["prev"], self.i_of_row, K)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, n):
        rows, P = self.seqs[n], self.pose
        rel = [rel_pose(P["R"][r], P["t"][r], P["R"][p], P["t"][p]) for p, r in zip(rows[:-1], rows[1:])]
        return dict(frames=[self.ds[self.i_of_row[r]] for r in rows], rel_R=torch.from_numpy(np.stack([R for R, _ in rel]).reshape(-1, 2, 2)), rel_t=torch.from_numpy(np.stack([t for _, t in rel]).reshape(-1, 2)))


def collate(b):
    K = len(b[0]["frames"])
    return dict(LD.collate([o["frames"][k] for k in range(K) for o in b]), K=K, rel_R=torch.stack([o["rel_R"] for o in b]), rel_t=torch.stack([o["rel_t"] for o in b]))


def loader(split="train", K=3, bs=4, workers=3, shuffle=True, map_mode="train", pixels=False):
    ds = SeqData(split, K=K, map_mode=map_mode, pixels=pixels)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle, persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)
