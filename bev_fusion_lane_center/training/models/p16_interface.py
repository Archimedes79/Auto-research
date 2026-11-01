"""Contract of the lane-graph builder: what `Builder.forward` reads and what it returns, written down once with names, shapes, units and frames, and checked.
Modelled on bev_fusion's training/interface.py (layers 2 and 4 of its five-layer design). Nothing in the live chain calls it yet: adding the call to the trainer and to
`p16_infer.predict` is the first line of step 0 of the E2E set-up and needs a critic pass because those files are shared.

    check_inputs(x)                  -> list of problems (empty = fine)
    check_outputs(res, cfg)          -> the same for the dict returned by Builder.forward
    python training/models/p16_interface.py     self-test: a tiny builder on random input satisfies the contract, damaged outputs are named

Frames and units: the ego frame of the current keyframe, metres, p0 = row (forward), p1 = column (left); angles in radians; the grid is 200 x 200 cells of 0.5 m (+-50 m)."""
import math

import torch

IN_CHANNELS = 42  # 39 frozen GaussianCaR channels (decoder PCA 32 + 7 map logits), z-scored; dense-head heat (1); dense-head direction field (2)

# name: (required, meaning) of the forward arguments
INPUTS = {
    "x": (True, "[B, 42, 200, 200] float: 39 normalised GaussianCaR channels + dense-head heat + dense-head direction (cos, sin); out-of-fold dense maps for training rows"),
    "gt": (False, "batch dict of ground-truth tensors (training only; read by the warm start `expert_state`, share > 0)"),
    "share": (False, "float in [0, 1]: share of the expert state mixed into the next tips (warm start, training only)"),
    "prior": (False, "dict pos [B,P,2], th [B,P], pi [B,P], h [B,P,d], valid [B,P]: the previous frame's best nodes warped to now (time recursion; made by p16_time.prior_from_prev)"),
}

# per node (N = 2 K0 seed tips + D levels of M nodes, concatenated along dim 1): name -> (dtype kind, trailing dims or None, meaning)
PER_NODE = {
    "pos": ("f", (2,), "node position, ego frame, metres"),
    "th_travel": ("f", (), "travel direction of the lane at the node, rad"),
    "flag": ("f", (), "+1 grown forward, -1 grown against the travel direction"),
    "nu": ("f", (), "node validity probability in [0, 1] (verification along the ray)"),
    "pi": ("f", (), "path validity from the seed to the node, in [0, 1] (bottleneck min of the node validities; times the keep gate when spf2)"),
    "valid": ("f", (), "1 for nodes the beam kept, 0 for padding"),
    "parent": ("l", (), "index of the parent node in the same concatenation, -1 for seed tips"),
    "level": ("f", (), "growth level, 0 = seed tip"),
    "h": ("f", None, "node token [d]"),
    "root": ("l", (), "index of the seed (0 .. K0-1) the node grows from"),
}
# always present besides the per-node arrays
STRUCT = ("levels", "seeds", "seed_idx", "seed_ok", "seed_logit", "dirh", "feats", "maps")
# optional outputs by switch of the config
OPTIONAL = {
    "junc": ("junc_logit",),  # [B, 2, H, W] split / merge heat logits
    "spf2": ("keep_logit", "z_keep", "p_open", "pi_tree"),
    "revisit": ("pos_rev", "keep_rev"),
    "link": ("link_ki", "link_logit", "link_mask", "link_off", "link_tconn", "link_cv"),  # constr L / H: [B, M, k] candidate links between forward and backward ends
}


def _opt(cfg):
    names = []
    if cfg.junc:
        names += OPTIONAL["junc"]
    if cfg.spf2:
        names += OPTIONAL["spf2"]
    if cfg.revisit:
        names += OPTIONAL["revisit"]
    if cfg.constr in ("L", "H"):
        names += OPTIONAL["link"]
    return names


def check_inputs(x):
    bad = []
    if not torch.is_tensor(x) or x.dim() != 4 or x.shape[1] != IN_CHANNELS:
        return [f"x must be [B, {IN_CHANNELS}, H, W], got {tuple(getattr(x, 'shape', ()))}"]
    if x.shape[2] != x.shape[3]:
        bad.append(f"x is not square: {tuple(x.shape)} (the builder samples +-50 m on a square grid)")
    if not bool(torch.isfinite(x.float()).all()):
        bad.append("x holds NaN or inf")
    return bad


def check_outputs(res, cfg):
    bad = []
    missing = [k for k in list(PER_NODE) + list(STRUCT) + _opt(cfg) if k not in res]
    if missing:
        return [f"missing outputs: {', '.join(missing)}"]
    B, N = res["pos"].shape[:2]
    for k, (kind, tail, _) in PER_NODE.items():
        t = res[k]
        if t.shape[:2] != (B, N):
            bad.append(f"{k}: leading shape {tuple(t.shape[:2])}, expected {(B, N)}")
            continue
        if tail is not None and tuple(t.shape[2:]) != tail:
            bad.append(f"{k}: trailing shape {tuple(t.shape[2:])}, expected {tail}")
        if kind == "l" and t.dtype not in (torch.long, torch.int32):
            bad.append(f"{k}: dtype {t.dtype}, expected an integer index")
        if kind == "f" and not bool(torch.isfinite(t.float()).all()):
            bad.append(f"{k}: NaN or inf")
    if bad:
        return bad
    K0 = res["seed_idx"].shape[1]
    if N != 2 * K0 + cfg.D * res["levels"][0]["sel"].shape[1] or len(res["levels"]) != cfg.D:
        bad.append(f"N = {N} does not match 2 K0 + D M (K0 {K0}, D {cfg.D})")
    for k in ("nu", "pi"):
        if float(res[k].min()) < -1e-4 or float(res[k].max()) > 1 + 1e-4:
            bad.append(f"{k} outside [0, 1]: {float(res[k].min()):.3f} .. {float(res[k].max()):.3f}")
    if not bool(((res["flag"].abs() - 1).abs() < 1e-4).all()):
        bad.append("flag holds values other than +-1")
    if int(res["parent"].min()) < -1 or int(res["parent"].max()) >= N:
        bad.append("parent index out of range")
    if not bool((res["parent"][:, : 2 * K0] == -1).all()):
        bad.append("a seed tip has a parent")
    if "link_ki" in res:
        M = N - int(res["link_off"])
        for k in ("link_ki", "link_logit", "link_mask"):
            if res[k].shape[:2] != (B, M):
                bad.append(f"{k}: leading shape {tuple(res[k].shape[:2])}, expected {(B, M)}")
        if res["link_logit"].shape != res["link_ki"].shape:
            bad.append("link_logit and link_ki differ in shape")
    if cfg.junc and (res["junc_logit"].dim() != 4 or res["junc_logit"].shape[1] != 2):
        bad.append(f"junc_logit must be [B, 2, H, W], got {tuple(res['junc_logit'].shape)}")
    return bad


def selftest():
    import p16_builder as PB

    torch.manual_seed(0)
    cfg = PB.DefaultCfg(constr="L", pattern="teardrop", seeds="lattice", ctx="global", junc=True, stage_c=True, b=3, M=16, K0=8, mcl=True)
    net = PB.Builder(cfg, stem_from=None).eval()
    x = torch.randn(1, IN_CHANNELS, 200, 200)
    assert check_inputs(x) == []
    assert check_inputs(torch.randn(1, 41, 200, 200)) and check_inputs(torch.full((1, 42, 8, 8), float("nan")))
    with torch.no_grad():
        res = net(x)
    assert check_outputs(res, cfg) == [], check_outputs(res, cfg)
    cut = {k: v for k, v in res.items() if k != "junc_logit"}
    assert any("junc_logit" in p for p in check_outputs(cut, cfg))
    broken = dict(res, pi=res["pi"] * 3.0)
    assert any("pi outside" in p for p in check_outputs(broken, cfg))
    broken = dict(res, parent=res["parent"].clone())
    broken["parent"][0, 0] = 5
    assert any("seed tip has a parent" in p for p in check_outputs(broken, cfg))
    broken = dict(res, pos=res["pos"] * float("nan"))
    assert any("NaN" in p for p in check_outputs(broken, cfg))
    print(f"p16_interface selftest passed (N = {res['pos'].shape[1]} nodes, {len(res['levels'])} level)")


if __name__ == "__main__":
    selftest()
