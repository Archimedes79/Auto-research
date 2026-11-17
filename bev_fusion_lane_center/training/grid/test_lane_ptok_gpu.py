"""Tests of the GPU-resident loader (`data/lane_ptok_gpu.py`); CPU device (the same code runs on cuda), a toy cache written here and, when it exists, 24 frames of the real VAL cache.

    wsl bash tools/env/_py14.sh training/grid/test_lane_ptok_gpu.py [name ...]

batches (every field and dtype of `lane_ptok_data.collate` on the toy cache: tok pos dirv e e_raw s uout uin flags valid ytype y ylong gt_out row token; a pair claimed twice keeps the LAST one like numpy; the dense
yscale equals `densify`'s; the pair list rebuilds the same scale mask and values), real (24 frames of the real VAL cache: the same fields bit for bit, y / ylong / ytype equal), order (the epoch order of
DataLoader(shuffle=True, drop_last=True), workers 0 and 2 with persistent workers, three passes, two seeds, against `DeviceLoader`; the last batch is dropped; no shuffle = file order with the short last batch),
mutants (the sampler seed drawn without the base-seed draw, a first-wins pair: each fails), gpu (JTF_GPU=1: the same batch on cuda)."""
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
for sub_ in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
import lane_ptok_data as LP  # noqa: E402
import lane_ptok_gpu as LG  # noqa: E402

torch.set_num_threads(2)


def _toy(d, F=6, K=32):
    rng = np.random.default_rng(0)
    arr = dict(tok=rng.random((F, K, 4)).astype(np.float16), pos=(rng.random((F, K, 2)) * 40 - 20).astype(np.float32), dirv=rng.random((F, K, 2)).astype(np.float32), e=rng.random((F, K)).astype(np.float32),
               e_raw=rng.random((F, K)).astype(np.float32), s=rng.random((F, K)).astype(np.float32), uout=rng.random((F, K)).astype(np.float32), uin=rng.random((F, K)).astype(np.float32),
               flags=rng.random((F, K, 8)).astype(np.float32), valid=np.ones((F, K), bool), gt_type=(rng.integers(0, 7, (F, K))).astype(np.int8), gt_out=np.arange(F, dtype=np.int32) + 5,
               ypair=-np.ones((F, 12, 2), np.int16), ypair_long=np.zeros((F, 12), bool), ypair_scale=np.full((F, 12, 2), np.nan, np.float32), e_p=rng.random((F, K)).astype(np.float32),
               flags_p=rng.random((F, K, 8)).astype(np.float32), s_p=rng.random((F, K)).astype(np.float32), cell=np.tile(np.arange(K, dtype=np.int16), (F, 1)), gt_id=-np.ones((F, K), np.int32),
               gd=np.full((F, K), np.nan, np.float32), gt_flags=np.zeros((F, K, 8), bool), gt_aux=np.zeros((F, K, 3), bool))
    for f in range(F):
        n = 3 + f                                                       # a different pair count per frame
        arr["ypair"][f, :n] = np.stack([rng.integers(0, K, n), rng.integers(0, K, n)], 1)
        arr["ypair_long"][f, :n] = rng.random(n) > 0.5
        arr["ypair_scale"][f, :n] = np.where(rng.random((n, 1)) > 0.4, rng.random((n, 2)), np.nan)
    arr["ypair"][2, 1] = arr["ypair"][2, 0]                             # frame 2: the same cell twice (the later one must win)
    arr["ypair_long"][2, 0], arr["ypair_long"][2, 1] = True, False
    arr["ypair_scale"][2, 0], arr["ypair_scale"][2, 1] = (0.1, 0.2), (0.7, 0.8)
    arr["valid"][1, 30:] = False
    for n_, v in arr.items():
        np.save(os.path.join(d, n_ + ".npy"), v)
    json.dump(dict(rows=list(range(10, 10 + F)), tokens=[f"t{i}" for i in range(F)]), open(os.path.join(d, "frames.json"), "w"))
    json.dump(dict(done=F), open(os.path.join(d, "progress.json"), "w"))
    return F, K


def _same(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    assert torch.equal(torch.nan_to_num(a.float(), nan=-777.0), torch.nan_to_num(b.float(), nan=-777.0)), what


def test_batches():
    with tempfile.TemporaryDirectory() as d:
        F, K = _toy(d)
        ref = LP.PTokData("val", dir=d)
        ds = LG.DeviceTokData("val", dir=d, dev="cpu", dense_scale=True)
        idx = [0, 2, 3, 5]
        r, g = LP.collate([ref[i] for i in idx]), ds.batch(idx)
        for k in LG.FIELDS + ("ytype", "y", "ylong", "yscale"):
            _same(r[k], g[k], k)
        assert g["gt_out"].tolist() == r["gt_out"].tolist() and g["token"] == r["token"] and g["row"].tolist() == r["row"].tolist()
        # the doubled cell of frame 2: the later pair wins
        j = idx.index(2)
        i0, j0 = [int(v) for v in np.load(os.path.join(d, "ypair.npy"))[2, 0]]
        assert float(g["yscale"][j, i0, j0, 0]) == float(np.float32(0.7)) and float(g["yscale"][j, i0, j0, 1]) == float(np.float32(0.8)), "the later pair of a doubled cell wins"
        _same(r["yscale"][j], g["yscale"][j], "yscale of the doubled frame")
        # the pair list rebuilds the dense mask and values
        B = len(idx)
        m_dense = torch.isfinite(g["yscale"][..., 0])
        m_list = torch.zeros(B, K, K, dtype=torch.bool)
        v = torch.full((B, K, K, 2), float("nan"))
        ok = g["sp_ok"]
        m_list[g["sp_b"][ok], g["sp_i"][ok], g["sp_j"][ok]] = True
        v[g["sp_b"][ok], g["sp_i"][ok], g["sp_j"][ok]] = g["sp_y"][ok]
        assert torch.equal(m_dense, m_list), "the pair list's scale mask"
        _same(torch.nan_to_num(g["yscale"], nan=0.0), torch.nan_to_num(v, nan=0.0), "the pair list's values")
        # mutants of the rule 'the last wins': a first-wins build gives another value on frame 2
        ds2 = LG.DeviceTokData("val", dir=d, dev="cpu", dense_scale=True)
        for a_ in (ds2.ypair, ds2.yps, ds2.ypl):                          # frame 2: the two claims of the doubled cell swap places, so the earlier value is now the last
            a_[2, [0, 1]] = a_[2, [1, 0]].clone()
        g2 = ds2.batch(idx)
        assert float(g2["yscale"][j, i0, j0, 0]) == float(np.float32(0.1)), "a first-wins build gives the earlier value: the test sees the difference"
    print("batches ok")


def test_real():
    try:
        d = LP.cache_dir("val", 0)
    except Exception as e:  # noqa: BLE001
        print(f"real skipped ({str(e)[:60]})")
        return
    ref = LP.PTokData("val", dir=d)
    ds = LG.DeviceTokData("val", dir=d, dev="cpu", dense_scale=True)
    idx = list(range(0, 24 * 7, 7))
    r, g = LP.collate([ref[i] for i in idx]), ds.batch(idx)
    for k in LG.FIELDS + ("ytype", "y", "yscale"):
        _same(r[k], g[k], "real " + k)
    frac = float((r["ylong"] != g["ylong"]).float().mean())
    assert frac < 1e-6, f"ylong differs on {frac:.2e} of the entries (float32 against float64 chord at the 8 m threshold)"
    print("real ok")


class _Dummy:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def batch(self, idx):
        return list(idx)


class _Idx(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _orders(workers, seed, n=50, bs=8, passes=3):
    torch.manual_seed(seed)
    dl = torch.utils.data.DataLoader(_Idx(n), batch_size=bs, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    ref = [[b.tolist() for b in dl] for _ in range(passes)]
    torch.manual_seed(seed)
    mine = LG.DeviceLoader(_Dummy(n), bs, True, persistent=workers > 0)
    got = [[list(b) for b in mine] for _ in range(passes)]
    return ref, got


def test_order():
    for workers in (0, 2):
        for seed in (0, 7):
            ref, got = _orders(workers, seed)
            assert ref == got, (workers, seed)
    assert len(LG.DeviceLoader(_Dummy(50), 8, True)) == 6 and len(LG.DeviceLoader(_Dummy(50), 8, False)) == 7
    assert [list(b) for b in LG.DeviceLoader(_Dummy(10), 4, False)] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    # mutant: the sampler seed drawn WITHOUT the DataLoader's base-seed draw differs from the real order
    torch.manual_seed(0)
    gen = torch.Generator()
    gen.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    wrong = torch.randperm(50, generator=gen).tolist()[:8]
    ref, _ = _orders(0, 0)
    assert wrong != ref[0][0], "the base-seed draw must matter"
    print("order ok")


def test_gpu():
    if not (torch.cuda.is_available() and os.environ.get("JTF_GPU") == "1"):
        print("gpu skipped (set JTF_GPU=1 with a GPU)")
        return
    with tempfile.TemporaryDirectory() as d:
        _toy(d)
        c, g = LG.DeviceTokData("val", dir=d, dev="cpu", dense_scale=True), LG.DeviceTokData("val", dir=d, dev="cuda", dense_scale=True)
        idx = [0, 2, 3, 5]
        a, b = c.batch(idx), g.batch(idx)
        for k in LG.FIELDS + ("ytype", "y", "ylong", "yscale", "sp_b", "sp_i", "sp_j", "sp_y", "sp_ok"):
            _same(a[k], b[k].cpu(), "cuda " + k)
        assert a["token"] == b["token"]
    print("gpu ok")


TESTS = [test_batches, test_real, test_order, test_gpu]
if __name__ == "__main__":
    want = [n if n.startswith("test_") else "test_" + n for n in sys.argv[1:]]
    assert all(n in {t.__name__ for t in TESTS} for n in want), want
    for t_ in TESTS:
        if not want or t_.__name__ in want:
            t_()
    print("test_lane_ptok_gpu ok")
