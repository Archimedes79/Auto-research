"""Test of the foundation extraction (`net/prep/foundation/gcar_features.py`, `check_features.py`) without a GPU, the foundation's code, its
weights or any data: the recipe files in the repository (the PCA basis and the sample list by their hashes), the projection and the layout turn
on known answers, the frame rule and the comparison of the check, the writer (a new part, an interrupted part goes on, a finished part is
refused), the messages when the clone, the weights or the raw data are missing, the command lines. CPU, seconds.
Run: `research tests lane_center --only test_foundation` (or python net/tests/test_foundation.py from the problem folder)."""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "net", "prep", "foundation"))
import check_features as CF  # noqa: E402
import gcar_features as GF  # noqa: E402


def missing(call, *words):
    """the call must raise Missing with every word in its text"""
    try:
        call()
    except GF.Missing as e:
        assert all(w in str(e) for w in words), f"{e} lacks one of {words}"
        return
    raise AssertionError(f"accepted although it must be refused ({words})")


def test_recipe():
    mean, base = GF.basis()                                           # the file's own hash is checked inside
    assert hashlib.sha256(mean.tobytes() + base.tobytes()).hexdigest() == "2d8713b94493ff28bb4473fa3cfb6d275357deeb28a2b7457af1b95684886fd1"
    tokens = GF.sample_tokens()
    assert [len(tokens[p]) for p in ("train", "val")] == [8414, 1706] and not set(tokens["train"]) & set(tokens["val"])
    for part, sha in GF.SAMPLES_SHA256.items():                       # the bytes the stored token files have
        assert hashlib.sha256(json.dumps(tokens[part]).encode()).hexdigest() == sha, part


def test_project_and_layout():
    mean, base = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    x = torch.zeros(3, 2, 2)
    x[:, 0, 1] = torch.tensor([2.0, 4.0, 3.0])                        # minus the mean: (1, 2, 0) -> (1, 4)
    p = GF.project(x, mean, base)
    assert p[:, 0, 1].tolist() == [1.0, 4.0] and p[:, 0, 0].tolist() == [-4.0, -7.0], p
    grid = torch.arange(6.0).reshape(2, 3)                            # network layout [H 2, W 3] -> lane layout [3, 2]: out[i, j] = in[H - 1 - j, i]
    assert GF.to_lane_layout(grid).tolist() == [[3.0, 0.0], [4.0, 1.0], [5.0, 2.0]]


def test_pick_rows():
    scene = np.repeat(np.arange(5), [4, 3, 1, 6, 5])                  # rows 0-3 | 4-6 | 7 | 8-13 | 14-18
    every = range(len(scene))
    assert CF.pick_rows(scene, every, 3) == [2, 7, 16]                # scenes 0, 2, 4: their middle rows already alternate even, odd, even
    assert CF.pick_rows(scene, every, 2) == [2, 17]                   # scenes 0, 4: the middle row 16 moves to the odd 17
    assert CF.pick_rows(scene, every, 5) == [2, 5, 7, 11, 16]         # a scene of one row keeps it (7, where an even row is due)
    assert CF.pick_rows(scene, every, 9) == CF.pick_rows(scene, every, 5) and CF.pick_rows(scene, every, 1) == [2]
    assert CF.pick_rows(scene, [18, 4, 5, 15, 14], 5) == [4, 15]      # only the rows that may be read: scenes 1 (4, 5: 5 moves to 4) and 4 (14, 15, 18)
    may = CF.open_rows()                                              # the TEST rows of the part "val" are closed to the check
    with open(os.path.join(ROOT, "data", "splits.json"), encoding="utf-8") as f:
        test = set(json.load(f)["test"])
    assert test and not test & set(may["val"]) and len(may["val"]) == 873 and len(may["train"]) == 8414
    assert not test & set(CF.pick_rows(np.arange(1706) // 40, may["val"], 30))
    assert [CF.row_set(p, r) for p, r in (("val", 3), ("train", 3), ("train", 4))] == ["val", "train odd rows", "train even rows"]


def test_compare():
    ref = np.arange(8, dtype=np.float16).reshape(2, 2, 2)
    same = CF.compare(ref.copy(), ref)
    assert same["equal"] and same["max_abs"] == 0 and same["mean_abs"] == 0 and abs(same["corr"] - 1) < 1e-12 and same["ref_mean_abs"] == 3.5
    new = ref.copy()
    new[0, 0, 0] += 2                                                 # one of eight values off by 2
    off = CF.compare(new, ref)
    assert not off["equal"] and off["max_abs"] == 2 and off["mean_abs"] == 0.25 and 0 < off["corr"] < 1
    assert abs(CF.compare(-ref, ref)["corr"] + 1) < 1e-12 and CF.compare(np.zeros_like(ref), ref)["corr"] is None
    s = CF.summarise([same, off])
    assert (s["frames"], s["equal"], s["max_abs"], s["mean_abs"]) == (2, 1, 2, 0.125) and s["corr_min"] == off["corr"]


def test_writer(tmp):
    tokens = [f"t{i}" for i in range(5)]
    rows = {t: {g: np.full((c, 200, 200), i + 0.5 * (g == "logit7"), np.float16) for g, c in GF.GROUPS.items()} for i, t in enumerate(tokens)}
    asked = []

    def source(toks, stop=None):
        asked.append(list(toks))
        for k, t in enumerate(toks):
            if k == stop:
                raise KeyboardInterrupt
            yield rows[t]

    out = os.path.join(tmp, "feats")
    try:
        GF.write_part(out, "val", tokens, lambda toks: source(toks, stop=3), every=2)
    except KeyboardInterrupt:
        pass
    assert open(os.path.join(out, "val.extracting")).read() == "2" and not os.path.exists(os.path.join(out, "val_tokens.json"))
    GF.write_part(out, "val", tokens, source, every=2)                # goes on from the row the mark names
    assert asked[-1] == tokens[2:] and not os.path.exists(os.path.join(out, "val.extracting"))
    for g, c in GF.GROUPS.items():
        a = np.load(os.path.join(out, f"val_{g}.npy"))
        assert a.shape == (5, c, 200, 200) and a.dtype == np.float16 and a[:, 0, 0, 0].tolist() == [i + 0.5 * (g == "logit7") for i in range(5)]
    assert open(os.path.join(out, "val_tokens.json")).read() == json.dumps(tokens)
    before = len(asked)
    missing(lambda: GF.write_part(out, "val", tokens, source), "never overwritten")
    assert len(asked) == before and np.load(os.path.join(out, "val_dec32.npy"))[4, 0, 0, 0] == 4


def test_missing(tmp):
    missing(lambda: GF.need_clone(None), "GCAR_REPO", "--clone", GF.REMOTE)
    missing(lambda: GF.need_clone(os.path.join(tmp, "nowhere")), "no GaussianCaR clone", GF.COMMIT)
    clone = os.path.join(tmp, "clone")
    os.makedirs(os.path.join(clone, "configs"))
    os.makedirs(os.path.join(clone, GF.RASTERISER))
    missing(lambda: GF.need_clone(clone), "is at commit", GF.COMMIT[:12])
    small = {"name": "w.ckpt", "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest(), "url": "https://example.invalid/w.ckpt"}
    missing(lambda: GF.need_weights(None), "GCAR_WEIGHTS", "--weights", GF.WEIGHTS["sha256"], GF.WEIGHTS["url"])
    path = os.path.join(tmp, "w.ckpt")
    missing(lambda: GF.need_weights(path, small), "no weights at", small["sha256"], small["url"])
    for content, word in ((b"abcd", "4 bytes"), (b"abd", "has sha256")):
        with open(path, "wb") as f:
            f.write(content)
        missing(lambda: GF.need_weights(path, small), word, small["sha256"])
    with open(path, "wb") as f:
        f.write(b"abc")
    assert GF.need_weights(path, small) == path
    missing(lambda: GF.need_raw(os.path.join(tmp, "nuscenes")), "raw nuScenes", "sample_data.json", "datasets/nuscenes/README.md")
    missing(lambda: CF.scratch(os.path.join(os.path.dirname(GF.BP.DATA), "lane_center", "x.json")), "below the data root")
    assert CF.scratch(os.path.join(tmp, "x.json")) == os.path.join(tmp, "x.json")


def test_command_lines():
    a = GF.parse(["--out", "/x", "--clone", "/c", "--weights", "/w"])
    assert (a.out, a.parts, a.clone, a.weights, a.workers) == ("/x", ["val", "train"], "/c", "/w", 2)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            GF.parse([])
    except SystemExit:
        pass
    else:
        raise AssertionError("gcar_features accepted a call without --out")
    c = CF.parse(["--parts", "train", "--clone", "/c"])
    assert (c.n, c.parts, c.clone, c.workers) == (8, ["train"], "/c", 0) and c.out.startswith(GF.BP.SCRATCH)


def main():
    test_recipe()
    test_project_and_layout()
    test_pick_rows()
    test_compare()
    with tempfile.TemporaryDirectory() as tmp:
        test_writer(tmp)
        test_missing(tmp)
    test_command_lines()
    print("test_foundation passed: recipe hashes, projection and layout, frame rule (no TEST row), comparison, writer (new, interrupted, refused), "
          "missing pieces, command lines")


if __name__ == "__main__":
    main()
