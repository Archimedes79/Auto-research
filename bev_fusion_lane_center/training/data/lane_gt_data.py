"""Data of the lane head (E273): the 39 frozen BEV channels of a frame + everything the heads are trained on, drawn from the lane GT of GtGenerator (`GtGenerator.lane_gt`) in the data loader.

    ds, dl = loader("train" | "val_clean", bs=4, workers=3, map_mode="train" | "error" | "exact")

Frames = the project's splits through `BuilderData` (TRAIN 8414, VAL 873 via `BuilderData.rows`: the stored feature files also hold the TEST rows, which are never iterated); the sample token of
a row is `~/exp_gcar_b123/feats/{train,val}_tokens.json[row]`; the lane GT of the token comes from the store of the lane GT of this checkout (`GtGenerator.lane_gt.load`). A frame whose graph is not
stored (yet) is left out of the dataset (`missing` says how many; `strict=True` raises instead).
Item (numpy -> torch): x (39, 200, 200) float16 (normalised as `GridData`); node targets `qtargets.grid_targets` on the 50 x 50 grid of 2 m cells: exists, flags (8), offset (m), dirv, sstart,
piece (int, index into the GT pieces of the frame, -1 none), uout, uin; links `lk` (L, 4) int cells (iu, ju, iv, jv) + `lk_long` + `lk_scale` (L, 2) the fitted Hermite handle scales of the long links (NaN for short ones); pixel targets `qraster.pixel_targets` (lines_mask / lines_dist
(3, 200, 200), g05 / t05 for the 1 m centre loss, events1 (6, 100, 100), road2 (2, 50, 50); the intersection polygons are grown by `dilate` = 2 m (Mirko: slightly enlarged; the GT itself is undilated: stop / exit events lie 1-3 m before the polygon edge)); the map `map_sim.make_map` with its piece targets (`piece_map`: GT piece -> map pieces,
`piece_road`).
`collate` stacks the fixed arrays and pads the variable ones (links with -1, g05 with 1e4, the map to the largest K / T of the batch, with masks `kmask`, `tmask`).
Run the self-test: python lane_gt_data.py
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import map_sim as MS  # noqa: E402
import p16_data as PD  # noqa: E402
import qends as QEND  # noqa: E402
import qtargets as QT  # noqa: E402
import qzones as QZ  # noqa: E402

BOX = 48.0
ZONE_NAMES = ("zone", "zone_kind", "zone_gt", "exists_dense", "offset_dense", "dirv_dense")          # E282: the maps of `qzones.zone_maps` (only with zone=True)


def lane_store():
    from GtGenerator import lane_gt
    return lane_gt, os.path.join(lane_gt.DATA, lane_gt.signature())


class QData(torch.utils.data.Dataset):
    DEFAULT_SPARSE = None                                          # E276: set by `qgraph_train.py --sparse EPS CHORD` so that the train loader AND the gates (they build their own QData) see the same targets
    DEFAULT_ZONE = False                                           # E282: set by `qgraph_train.py --node-loss zone`: the batch also carries the zone maps (needs a sparse target)
    DEFAULT_ENDS = False                                           # E284: set by `qgraph_train.py --phases 8`: the batch also carries the junction ends and the true pairs (`qends.end_table`)

    def __init__(self, split="train", ks=None, map_mode="train", pixels=True, strict=False, dilate=2.0, sparse=None, zone=None, ends=None):
        self.bd = PD.BuilderData(split, ks=ks, train=(split == "train"))
        self.map_mode, self.pixels, self.dilate = map_mode, pixels, dilate
        sparse = QData.DEFAULT_SPARSE if sparse is None else sparse
        self.sparse = None if sparse is None else (float(sparse[0]), float(sparse[1]))                   # E276: the sparse node target (eps, max chord) in metres, None = a node in every 2 m cell
        self.zone = QData.DEFAULT_ZONE if zone is None else bool(zone)
        self.ends = QData.DEFAULT_ENDS if ends is None else bool(ends)
        assert not self.zone or self.sparse is not None, "the zone maps need the sparse node target"
        part = "train" if split == "train" else "val"
        self.tokens = json.load(open(f"{PD.D}/feats/{part}_tokens.json"))
        self.lg, d = lane_store()
        held = self.lg.held(d)
        ks_all = [int(k) for k in self.bd.ks]
        self.ks = [k for k in ks_all if self.tokens[int(self.bd.rows[k])] in held]
        self.missing = len(ks_all) - len(self.ks)
        if strict and self.missing:
            raise FileNotFoundError(f"{self.missing} of {len(ks_all)} frames have no stored lane GT in {d}")
        self.mm = None

    def __len__(self):
        return len(self.ks)

    def _open(self):
        if self.mm is None:
            self.mm = [np.load(f"{PD.D}/feats/{self.bd.part}_{g}.npy", mmap_mode="r") for g in ("dec32", "logit7")]

    def __getitem__(self, i):
        self._open()
        k = self.ks[i]
        row = int(self.bd.rows[k])
        token = self.tokens[row]
        x = np.concatenate([np.asarray(m[row], np.float32) for m in self.mm], 0)
        x = ((x - self.bd.mean[:39, None, None]) / self.bd.std[:39, None, None]).astype(np.float16)
        g = self.lg.load(token)
        nt = QT.node_table(g, sparse=self.sparse)
        t = QT.grid_targets(nt)
        lk, lk_long = QT.link_array(nt)
        lk_scale = QT.fit_long_scales(nt).astype(np.float32)
        mp = MS.make_map(g, self.map_mode, rng=np.random.RandomState() if self.map_mode == "train" else None, token=token)
        out = dict(x=torch.from_numpy(x), row=row, k=k, token=token, lk=torch.from_numpy(lk), lk_long=torch.from_numpy(lk_long), lk_scale=torch.from_numpy(lk_scale),
                   piece_map=MS.piece_targets(g, mp), piece_road=MS.road_targets(g, mp), mp=mp, n_pieces=len(g["pieces"]))
        out.update({n: torch.from_numpy(v) for n, v in t.items()})
        if self.zone:                                                                                  # E282: zones of the sparse target + the DENSE offset / direction of every cell's node
            out.update({n: torch.from_numpy(np.ascontiguousarray(v)) for n, v in QZ.zone_maps(QZ.zones(g, self.sparse)).items()})
        if self.ends:                                                                                 # E284: the GT ends of the junctions (lane frame x, y, tx, ty) and the true (entry, exit) pairs
            et = QEND.end_table(g) or QEND.empty_table()
            out.update(ent=torch.from_numpy(et["ent"]).float().reshape(-1, 4), ext=torch.from_numpy(et["ext"]).float().reshape(-1, 4), pairs=torch.from_numpy(et["pairs"]).long().reshape(-1, 2))
        if self.pixels:
            import qraster as QR
            pix = QR.pixel_targets(g, dilate=self.dilate)
            out.update({n: torch.from_numpy(np.asarray(pix[n])) for n in ("lines_mask", "lines_dist", "g05", "t05", "events1", "road2")})
        return out


def collate(b):
    B = len(b)
    out = dict(row=torch.tensor([o["row"] for o in b]), k=torch.tensor([o["k"] for o in b]), token=[o["token"] for o in b])
    for n in ("x", "exists", "flags", "offset", "dirv", "sstart", "piece", "uout", "uin", "tier", "vis", "ztier"):        # tier / vis / ztier: E276 (l4 labels, zeros / ones without them)
        out[n] = torch.stack([o[n] for o in b])
    for n in ("lines_mask", "lines_dist", "events1", "road2") + ZONE_NAMES:
        if n in b[0]:
            out[n] = torch.stack([o[n] for o in b])
    if "g05" in b[0]:
        G = max(1, max(len(o["g05"]) for o in b))
        g05, t05, m05 = torch.full((B, G, 2), 1e4), torch.zeros(B, G, 2), torch.zeros(B, G)
        for i, o in enumerate(b):
            n = len(o["g05"])
            g05[i, :n], t05[i, :n], m05[i, :n] = o["g05"], o["t05"], 1.0
        out.update(g05=g05, t05=t05, m05=m05)
    if "ent" in b[0]:                                                                                  # E284: ends and true pairs, padded (masks ent_m / ext_m, pairs -1)
        E, X, P_ = (max(1, max(len(o[k]) for o in b)) for k in ("ent", "ext", "pairs"))
        ent, ext, pairs = torch.zeros(B, E, 4), torch.zeros(B, X, 4), -torch.ones(B, P_, 2, dtype=torch.long)
        ent_m, ext_m = torch.zeros(B, E, dtype=torch.bool), torch.zeros(B, X, dtype=torch.bool)
        for i, o in enumerate(b):
            ent[i, :len(o["ent"])], ext[i, :len(o["ext"])], pairs[i, :len(o["pairs"])] = o["ent"], o["ext"], o["pairs"]
            ent_m[i, :len(o["ent"])], ext_m[i, :len(o["ext"])] = True, True
        out.update(ent=ent, ext=ext, ent_m=ent_m, ext_m=ext_m, pairs=pairs)
    L = max(1, max(len(o["lk"]) for o in b))
    lk, lkl, lks = -torch.ones(B, L, 4, dtype=torch.long), torch.zeros(B, L, dtype=torch.bool), torch.full((B, L, 2), float("nan"))
    for i, o in enumerate(b):
        n = len(o["lk"])
        lk[i, :n], lkl[i, :n], lks[i, :n] = o["lk"], o["lk_long"], o["lk_scale"]
    out.update(lk=lk, lk_long=lkl, lk_scale=lks)
    K = max(1, max(o["mp"]["K"] for o in b))
    T = max(1, max(len(o["mp"]["pos"]) for o in b))
    P = max(1, max(o["n_pieces"] for o in b))
    pos, tan, frac, pidx = torch.zeros(B, T, 2), torch.zeros(B, T, 2), torch.zeros(B, T), -torch.ones(B, T, dtype=torch.long)
    pfeat, adj, kmask, road = torch.zeros(B, K, 12), torch.zeros(B, 5, K, K, dtype=torch.bool), torch.zeros(B, K, dtype=torch.bool), -torch.ones(B, K, dtype=torch.long)
    piece_map, piece_road = torch.zeros(B, P, K, dtype=torch.bool), -torch.ones(B, P, dtype=torch.long)
    for i, o in enumerate(b):
        m = o["mp"]
        t, k_ = len(m["pos"]), m["K"]
        pos[i, :t], tan[i, :t], frac[i, :t], pidx[i, :t] = torch.from_numpy(m["pos"]), torch.from_numpy(m["tan"]), torch.from_numpy(m["frac"]), torch.from_numpy(m["pidx"])
        pfeat[i, :k_], adj[i, :, :k_, :k_], kmask[i, :k_], road[i, :k_] = torch.from_numpy(m["pfeat"]), torch.from_numpy(m["adj"]), True, torch.from_numpy(m["road"])
        for p, ks in enumerate(o["piece_map"]):
            for kk in ks:
                piece_map[i, p, kk] = True
            piece_road[i, p] = o["piece_road"][p]
    out.update(map_ids=[o["mp"]["ids"] for o in b], map_roads=[o["mp"]["roads"] for o in b], map_pos=pos, map_tan=tan, map_frac=frac, map_pidx=pidx, map_pfeat=pfeat, map_adj=adj, kmask=kmask, map_road=road, piece_map=piece_map, piece_road=piece_road,
               tmask=pidx >= 0)
    return out


def loader(split="train", bs=4, workers=3, shuffle=True, ks=None, map_mode="train", pixels=True, strict=False, sparse=None, zone=None, ends=None):
    ds = QData(split, ks=ks, map_mode=map_mode, pixels=pixels, strict=strict, sparse=sparse, zone=zone, ends=ends)
    return ds, torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, collate_fn=collate, drop_last=shuffle,
                                           persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None)


def _selftest():
    ds, dl = loader("val_clean", bs=2, workers=0, shuffle=False, pixels=False, map_mode="error")
    print("VAL frames with stored lane GT:", len(ds), "of", len(ds) + ds.missing)
    assert len(ds) > 0, "no stored lane GT frame in the VAL split yet"
    b = next(iter(dl))
    assert b["x"].shape == (2, 39, 200, 200) and b["exists"].shape == (2, 50, 50) and b["flags"].shape == (2, 8, 50, 50)
    assert (b["piece"][b["exists"] > 0] >= 0).all() and (b["piece"][b["exists"] == 0] == -1).all()
    assert b["lk"].shape[0] == 2 and b["piece_map"].shape[0] == 2
    # a node's piece has a map piece in the exact map: every existing node maps to at least one map piece (mode error drops 5 % of the pieces)
    print("batch ok: nodes", int(b["exists"].sum()), "links", int((b["lk"][..., 0] >= 0).sum()), "map pieces", b["kmask"].sum(1).tolist())
    print("lane_gt_data selftest ok")


if __name__ == "__main__":
    _selftest()
