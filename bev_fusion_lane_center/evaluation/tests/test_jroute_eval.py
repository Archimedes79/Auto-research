"""Tests of `jroute_eval.py` (E389 read-out of the SD-route association head): CPU only, no GPU.

    wsl bash tools/env/_py14.sh test_jroute_eval.py            # the selftest, the textual mutants, then dump / score / rules on the first frames of the real VAL cache with fresh heads
    wsl bash tools/env/_py14.sh test_jroute_eval.py selftests  # the selftest and the real-cache run without the mutants
    wsl bash tools/env/_py14.sh test_jroute_eval.py only "bootstrap" "padding"   # those mutants alone (a part of the name); `only all` = all mutants without the real-cache run

Selftest (`jroute_eval.py selftest`): hand frames with the answers by hand for every number (F1 / P / R at both levels, the fork frames, route success@30 m, the false tokens, the calibration bins, a frame drawn twice), the delivered
definitions at their boundaries, the fork clause at its minimum, the paired bootstrap (identical systems differ by exactly 0, a draw is whole scenes), the dump writer (NaN at a padding token), the straight prior and the geometric rule by hand,
a card is never overwritten. A mutant = one textual change of the source (a threshold moved, padding or undelivered or unlabelled tokens counted, fork frames mixed with every frame, the bootstrap unpaired or by frame, ...); the mutated copy runs
`selftest` in a temp folder and must fail by an assertion; a mutant that does not apply counts as failed.
Real cache (skipped when the route cache of the data layer, the token cache or the champion's dump is not there): fresh random heads (arm m, arm z) in a temp runs folder -> `dump` in several modes, `score` (single, paired, paired across
modes), `rules`: the dump equals a direct forward of the head (padding NaN), arm z reads no route (the same probabilities in every mode), arm m does, a dump / card / rules card is never overwritten, a dump of another model.pt is refused, the card's
F1 / fork F1 / success30 equal an independent per-token python loop over the caches, the paired difference is the difference of the two single cards, a self-pair is exactly 0, flag-all is 2p / (1 + p) of the label share.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.join(HERE, "..", "jroute_eval.py")
ENV = dict(os.environ, BFLC_ROOT=ROOT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

MUTANTS = [
    ("flag at p > 0.5", "return np.asarray(p) >= THR", "return np.asarray(p) > THR"),
    ("flag at p >= 0.4", "THR = 0.5 ", "THR = 0.4 "),
    ("undelivered covered nodes scored", 'cov = delivered & side["valid"] & (side["gt_id"] >= 0) & has\n', 'cov = side["valid"] & (side["gt_id"] >= 0) & has\n'),
    ("padding tokens not masked", 'cov = delivered & side["valid"] & (side["gt_id"] >= 0) & has\n', 'cov = delivered & (side["gt_id"] >= 0) & has\n'),
    ("tokens without a GT node scored as nodes", 'cov = delivered & side["valid"] & (side["gt_id"] >= 0) & has\n', 'cov = delivered & side["valid"] & has\n'),
    ("frames without a route scored", 'cov = delivered & side["valid"] & (side["gt_id"] >= 0) & has\n', 'cov = delivered & side["valid"] & (side["gt_id"] >= 0)\n'),
    ("false tokens: undelivered ones counted", 'fal = delivered & side["valid"] & (side["gt_id"] < 0) & has', 'fal = side["valid"] & (side["gt_id"] < 0) & has'),
    ("fork frames = every frame with a route", 'fork=side["decision"] & side["has_route"]', 'fork=side["has_route"]'),
    ("F1 without the factor 2", "return div(2 * tp, 2 * tp + fp + fn)", "return div(tp, tp + fp + fn)"),
    ("success: two route nodes are enough", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.9, 0.2", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 2, 0.9, 0.2"),
    ("success window 45 m", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.9, 0.2", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 45.0, 3, 0.9, 0.2"),
    ("success: half of the route nodes flagged is enough", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.9, 0.2", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.5, 0.2"),
    ("success: 70 % of the non-route nodes may be flagged", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.9, 0.2", "SUCC_M, SUCC_MIN, SUCC_FLAG, SUCC_FALSE = 30.0, 3, 0.9, 0.7"),
    ("success counts nodes behind the vehicle", '(side["pos"][..., 1] >= 0) & ', ""),
    ("kept: p_none <= 0.6", "< np.float32(keep_thr))", "<= np.float32(keep_thr))"),
    ("kept: padding tokens kept", "return np.asarray(valid, bool) & (np.asarray(pnone, np.float32) < np.float32(keep_thr))", "return (np.asarray(pnone, np.float32) < np.float32(keep_thr))"),
    ("kept: the record threshold 0.5", "KEEP_THR = 0.6 ", "KEEP_THR = 0.5 "),
    ("proxy: e_p > 0.5", "(np.asarray(e_p) >= thr)", "(np.asarray(e_p) > thr)"),
    ("bootstrap unpaired (the reference on other draws)", "lambda sel: numbers(Cb, sel)", "lambda sel: numbers(Cb, None if sel is None else sel[:len(sel) // 2].repeat(2))"),
    ("bootstrap by frame, not by scene", "ST.scene_draws(np.asarray(scene), B, 0)", "ST.scene_draws(np.arange(len(scene)), B, 0)"),
    ("padding probability kept in the dump", "p_road=np.where(valid[i], p_road[i], np.nan).astype(np.float32)", "p_road=np.asarray(p_road[i]).astype(np.float32)"),
    ("fork clause: > 60 frames", "ok = nf >= MIN_FORK_FRAMES and", "ok = nf > MIN_FORK_FRAMES and"),
    ("fork clause: > 8 scenes", "and ns >= MIN_FORK_SCENES", "and ns > MIN_FORK_SCENES"),
    ("straight prior: |x| <= 4.5", "STRAIGHT_X, COS_MIN = 3.5, 0.7", "STRAIGHT_X, COS_MIN = 4.5, 0.7"),
    ("straight prior: |x| < 3.5", "<= STRAIGHT_X)", "< STRAIGHT_X)"),
    ("straight prior: heading cosine 0.5", "STRAIGHT_X, COS_MIN = 3.5, 0.7", "STRAIGHT_X, COS_MIN = 3.5, 0.5"),
    ("straight prior: heading cosine > 0.7", '(side["dirv"][..., 1] >= COS_MIN)', '(side["dirv"][..., 1] > COS_MIN)'),
    ("geometric rule: half lane 2.75 m", "HALF_LANE, MARGIN, GAP = 1.75, 0.5, 2.5", "HALF_LANE, MARGIN, GAP = 2.75, 0.5, 2.5"),
    ("geometric rule: margin 1.5 m", "HALF_LANE, MARGIN, GAP = 1.75, 0.5, 2.5", "HALF_LANE, MARGIN, GAP = 1.75, 1.5, 2.5"),
    ("geometric rule: samples are points, not a polyline", "HALF_LANE, MARGIN, GAP = 1.75, 0.5, 2.5", "HALF_LANE, MARGIN, GAP = 1.75, 0.5, 0.5"),
    ("geometric rule: the count of the first end", "np.where(t[k, j] < 0.5, cnt[j, 0], cnt[j, 1])", "cnt[j, 0]"),
    ("geometric rule: the larger count of the two ends", "np.where(t[k, j] < 0.5, cnt[j, 0], cnt[j, 1])", "np.maximum(cnt[j, 0], cnt[j, 1])"),
    ("geometric rule: heading cosine > 0.7", "((dirv * d[j]).sum(-1) >= COS_MIN)", "((dirv * d[j]).sum(-1) > COS_MIN)"),
    ("geometric rule: no heading test", "& ((dirv * d[j]).sum(-1) >= COS_MIN)", ""),
    ("dump identity: route cap not compared", 'IDENT_KEYS = ("version", "lane_gt", "tokens_sha", "max_samples")', 'IDENT_KEYS = ("version", "lane_gt", "tokens_sha")'),
    ("dump identity: frame list not compared", 'IDENT_KEYS = ("version", "lane_gt", "tokens_sha", "max_samples")', 'IDENT_KEYS = ("version", "lane_gt", "max_samples")'),
    ("reference dump identity not checked", '    if r is not None:\n        check_route_cache(r["meta"]["route_cache"], now, "the reference dump")\n', ""),
    ("dump identity not checked", '    check_route_cache(d["meta"]["route_cache"], now, "the dump")\n', ""),
    ("trained-on: route cap not compared", 'TRAINED_KEYS = ("version", "lane_gt", "max_samples")', 'TRAINED_KEYS = ("version", "lane_gt")'),
    ("trained-on: version not compared", 'TRAINED_KEYS = ("version", "lane_gt", "max_samples")', 'TRAINED_KEYS = ("lane_gt", "max_samples")'),
    ("trained-on: lane GT not compared", 'TRAINED_KEYS = ("version", "lane_gt", "max_samples")', 'TRAINED_KEYS = ("version", "max_samples")'),
    ("trained-on: the TRAIN frame list compared", 'TRAINED_KEYS = ("version", "lane_gt", "max_samples")', 'TRAINED_KEYS = ("version", "lane_gt", "tokens_sha", "max_samples")'),
    ("a recorded route cache ignored", "    if got is not None:", "    if False:"),
    ("a head older than the cache accepted", 'assert model_mtime >= manifest_mtime, (f"', 'assert model_mtime > manifest_mtime - 99, (f"'),
    ("a head as old as the cache refused", 'assert model_mtime >= manifest_mtime, (f"', 'assert model_mtime > manifest_mtime, (f"'),
    ("gap closed against nothing", "return div(m - z, rule - z)", "return div(m - z, rule)"),
    ("a card is overwritten", '    assert not os.path.exists(path), f"a stored card is never overwritten: {path}"\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, "x") as fh:',
     '    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, "w") as fh:'),
]


def run(src, *args):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "jroute_eval.py")
        open(p, "w", encoding="utf-8").write(src)
        return subprocess.run([sys.executable, p, *args], capture_output=True, text=True, env=ENV, cwd=d, timeout=900)


def test_selftest():
    r = run(open(MODULE, encoding="utf-8").read(), "selftest")
    assert r.returncode == 0 and "jroute_eval selftest ok" in r.stdout, r.stdout + r.stderr[-1500:]
    print("jroute_eval: selftest ok")


def test_cost():
    """`cost`: the parameter counts of the two input sets (33,738 / 21,130, the head's pinned numbers), the FLOPs grow with the number of route samples, never overwritten (no data needed)"""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, os.path.join(HERE, ".."))
    for sub_ in ("training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train"):
        sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
    import jroute as JR
    import jroute_eval as E
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as runs:
        for name, inputs, attn in (("c_full64", "full", "ab"), ("c_full160", "full", "ab"), ("c_geo160", "geo", "ab"), ("c_a160", "full", "a"), ("c_old160", "full", None)):         # None = a run of before the option attn
            os.makedirs(os.path.join(runs, name))
            torch.manual_seed(0)
            torch.save(dict(args=dict(seed=0, init_seed=0, cache=None, ident="e292_a", K=None, inputs=inputs, arm="m", steps=0, **({} if attn is None else dict(attn=attn))), state=JR.RouteHead(inputs=inputs, attn=attn or "ab").state_dict()), os.path.join(runs, name, "model.pt"))
        got = {name: json.load(open(E.cost(E.parse_args(["cost", "--run", name, "--K", "512", "--M", str(m)]), runs))) for name, m in (("c_full64", 64), ("c_full160", 160), ("c_geo160", 160), ("c_a160", 160), ("c_old160", 160))}
        assert got["c_full64"]["params"] == got["c_full160"]["params"] == 33738 and got["c_geo160"]["params"] == 21130, [g["params"] for g in got.values()]
        assert got["c_a160"]["params"] == 33226 and got["c_a160"]["attn"] == "a" and got["c_full160"]["attn"] == "ab" and got["c_old160"]["attn"] == "ab" and got["c_old160"]["params"] == 33738, "the cost card names the attn (a run without it is ab)"
        assert 0 < got["c_full64"]["gmacs"] < got["c_full160"]["gmacs"] and got["c_geo160"]["gmacs"] < got["c_full160"]["gmacs"] and got["c_full64"]["M"] == 64 and got["c_full160"]["timed"] is False and got["c_full160"]["clause_6"].startswith("UNVERIFIED") and "NOT a timing" in got["c_full160"]["note"], got
        try:
            E.cost(E.parse_args(["cost", "--run", "c_full64", "--M", "64"]), runs)
            raise RuntimeError("a cost card was overwritten")
        except AssertionError:
            pass
    print(f"jroute_eval: cost ok ({got['c_full160']['gmacs']:.3f} GMACs at K 512, M 160)")


def test_mutants(only=()):
    src = open(MODULE, encoding="utf-8").read()
    survived = []
    todo = [m for m in MUTANTS if not only or only == ["all"] or any(o in m[0] for o in only)]
    for name, old, new in todo:
        if src.count(old) != 1:
            survived.append((name, f"the text to change occurs {src.count(old)} times"))
            continue
        r = run(src.replace(old, new), "selftest")
        if r.returncode == 0 or "AssertionError" not in r.stderr:
            survived.append((name, r.returncode, r.stderr[-200:]))
    assert not survived, f"mutants not killed by an assertion: {survived}"
    print(f"jroute_eval: {len(todo)} of {len(todo)} mutants killed")


# ------------------------------------------------------------------------------------------------------------------------------ the real cache
def reference_loop(side, deliv, p_road, p_lane):
    """an independent per-token python loop over the arrays of the caches: (road tp fp fn, lane tp fp fn, fork road tp fp fn, eligible frames, successes)"""
    import numpy as np
    F, K = side["valid"].shape
    road, lane, fork = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    elig = succ = 0
    for i in range(F):
        if not side["has_route"][i]:
            continue
        nr = nrf = nn = nnf = 0
        for k in range(K):
            if not (side["valid"][i, k] and deliv[i, k] and side["gt_id"][i, k] >= 0):
                continue
            yr = side["ds"].rc.a["on_road"][i, side["gt_id"][i, k]] == 1
            yl = side["ds"].rc.a["on_lane"][i, side["gt_id"][i, k]] == 1
            fr, fl = p_road[i, k] >= 0.5, p_lane[i, k] >= 0.5
            for cnt, y, f in ((road, yr, fr), (lane, yl, fl)) + (((fork, yr, fr),) if side["decision"][i] else ()):
                cnt[0] += int(y and f)
                cnt[1] += int(not y and f)
                cnt[2] += int(y and not f)
            x, yy = side["pos"][i, k]
            if yy >= 0 and np.hypot(x, yy) <= 30.0:
                nr, nrf, nn, nnf = nr + int(yr), nrf + int(yr and fr), nn + int(not yr), nnf + int(not yr and fr)
        if nr >= 3:
            elig += 1
            succ += int(nrf >= 0.9 * nr - 1e-9 and nnf <= 0.2 * nn + 1e-9)
    return road, lane, fork, elig, succ


def test_real_cache(n=24):
    import numpy as np
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, os.path.join(HERE, ".."))
    for sub_ in ("training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train"):
        sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
    import jroute as JR
    import jroute_eval as E
    import jroute_train as JRT
    import lane_route_data as LRD
    man = os.path.join(LRD.route_dir("val"), "manifest.json")
    if not (os.path.exists(man) and os.path.exists(os.path.join(ROOT, "runs", "e373_cb_s0", "token_val.pkl"))):
        print("jroute_eval: real-cache test skipped (no route cache or no champion dump)")
        return
    try:
        LRD.TokenRoute(LRD.route_dir("val"))
        side = E.load_side(0, n_max=n)
    except (AssertionError, FileNotFoundError) as e:
        print(f"jroute_eval: real-cache test skipped ({str(e)[:120]})")
        return
    torch.set_num_threads(2)
    args = E.parse_args
    with tempfile.TemporaryDirectory() as runs:
        for name, arm, seed in (("m_s0", "m", 0), ("z_s0", "z", 1)):
            torch.manual_seed(seed)
            os.makedirs(os.path.join(runs, name))
            torch.save(dict(args=dict(seed=0, init_seed=0, cache=None, ident="e292_a", K=None, inputs="full", arm=arm, steps=0), state=JR.RouteHead().state_dict()), os.path.join(runs, name, "model.pt"))
        for run_, mode in (("m_s0", "error"), ("m_s0", "none"), ("m_s0", "exact"), ("z_s0", "error"), ("z_s0", "none")):
            E.dump(args(["dump", "--run", run_, "--seed", "0", "--mode", mode, "--cpu", "--max", str(n)]), runs)
        try:
            E.dump(args(["dump", "--run", "m_s0", "--seed", "0", "--mode", "error", "--cpu", "--max", str(n)]), runs)
            raise RuntimeError("a stored dump was overwritten")
        except AssertionError:
            pass
        rd = lambda run_, mode: E.read_dump(runs, run_, 0, mode, n)  # noqa: E731
        m_err, m_none, m_ex, z_err, z_none = rd("m_s0", "error"), rd("m_s0", "none"), rd("m_s0", "exact"), rd("z_s0", "error"), rd("z_s0", "none")
        assert m_err["tokens"] == side["tokens"] and np.array_equal(m_err["valid"], side["valid"]) and m_err["meta"]["autocast"] is False and m_err["meta"]["device"] == "cpu" and m_err["meta"]["attn"] == "ab"
        assert np.isnan(m_err["p_road"][~side["valid"]]).all() and np.isfinite(m_err["p_road"][side["valid"]]).all() and m_err["p_road"].dtype == np.float32
        # arm z reads no route: the same probabilities in every mode; arm m does read it
        assert np.array_equal(z_err["p_road"], z_none["p_road"], equal_nan=True) and np.array_equal(z_err["p_lane"], z_none["p_lane"], equal_nan=True), "arm z depends on the map mode"
        assert not np.array_equal(m_ex["p_road"], m_none["p_road"], equal_nan=True) and not np.array_equal(m_err["p_road"], m_ex["p_road"], equal_nan=True), "arm m does not read the route"
        # the dump of arm m is a direct forward of the head on the loader's batches
        m, _ = JRT.load_model("m_s0", "cpu", runs)
        ds, dl = LRD.loader("val", 0, 8, 0, mode="error", shuffle=False)
        b = next(iter(dl))
        with torch.no_grad():
            out = m(JRT.to_device(b, "full", "cpu"))
        v = b["valid"].numpy()
        assert np.allclose(torch.sigmoid(out["road"]).numpy()[v], m_err["p_road"][:8][v], atol=1e-6) and np.allclose(torch.sigmoid(out["lane"]).numpy()[v], m_err["p_lane"][:8][v], atol=1e-6), "the dump is not the head's forward"
        # the card equals the independent loop (kept = the champion's D115 decode, and the proxy)
        path = E.score(args(["score", "--run", "m_s0", "--seed", "0", "--mode", "error", "--max", str(n), "--draws", "50"]), runs)
        card = __import__("json").load(open(path))
        pn, _ = E.champion_pnone(side, 0)
        deliv = E.delivered_sets(side, pn)
        for name in ("kept", "proxy"):
            road, lane, fork, elig, succ = reference_loop(side, deliv[name], m_err["p_road"], m_err["p_lane"])
            nm = card["by_delivered"][name]["numbers"]
            f1 = lambda c: 2 * c[0] / (2 * c[0] + c[1] + c[2])  # noqa: E731
            assert abs(nm["route_F1_road"] - f1(road)) < 1e-9 and abs(nm["route_F1_lane"] - f1(lane)) < 1e-9 and abs(nm["fork_F1_road"] - f1(fork)) < 1e-9, (name, nm, road, lane, fork)
            assert nm["eligible_frames"] == elig and nm["success_frames"] == succ, (name, nm["eligible_frames"], elig, nm["success_frames"], succ)
        assert card["attn"] == "ab" and card["autocast"] is False and card["delivered_used"] == "kept" and card["by_delivered"]["kept"]["numbers"]["nodes"] != card["by_delivered"]["proxy"]["numbers"]["nodes"] and card["data"]["route_cache"]["version"] == LRD.VERSION
        # the reference rows (kept before the pairs: the paired card reads them): flag-all is 2p / (1 + p) of the label share of the covered nodes, the geometric rule on a wrong or a shifted map is worse than on the exact one
        rc = __import__("json").load(open(E.rules(args(["rules", "--seed", "0", "--split", "val", "--max", str(n)]), runs)))
        for name in ("kept", "proxy"):
            cov = deliv[name] & side["valid"] & (side["gt_id"] >= 0)
            share = float((side["lab_road"][cov] == 1).mean())
            rows = rc["by_delivered"][name]["rows"]
            f1r = lambda row: rows[row]["numbers"]["route_F1_road"]  # noqa: E731
            assert abs(f1r("flag_all") - 2 * share / (1 + share)) < 1e-9 and set(rows) == {"flag_all", "straight_prior", "geometric_exact", "geometric_error", "geometric_gross", "geometric_wrong"}, (name, share)
            assert rows["flag_all"]["numbers"]["route_R_road"] == 1.0 and f1r("geometric_exact") >= f1r("geometric_error") - 0.05 and f1r("geometric_wrong") < f1r("geometric_exact") - 0.2, (name, [f1r(r) for r in rows])
        # a card is never overwritten; a pair: the paired difference is the difference of the two single cards, a self-pair is exactly 0, the gap closed against the geometric rule of the rules card
        try:
            E.score(args(["score", "--run", "m_s0", "--seed", "0", "--mode", "error", "--max", str(n), "--draws", "50"]), runs)
            raise RuntimeError("a stored card was overwritten")
        except AssertionError:
            pass
        E.score(args(["score", "--run", "z_s0", "--seed", "0", "--mode", "error", "--max", str(n), "--draws", "50"]), runs)
        pair = __import__("json").load(open(E.score(args(["score", "--run", "m_s0", "--seed", "0", "--mode", "error", "--ref-run", "z_s0", "--max", str(n), "--draws", "50"]), runs)))
        single_z = __import__("json").load(open(os.path.join(runs, "z_s0", f"route_card_error_n{n}.json")))
        for name in ("kept", "proxy"):
            b = pair["by_delivered"][name]
            for key in E.BOOT_KEYS:
                d = card["by_delivered"][name]["numbers"][key] - single_z["by_delivered"][name]["numbers"][key]
                if d == d:
                    assert abs(b["paired"][key]["diff"] - d) < 1e-9, (name, key)
        for name in ("kept", "proxy"):
            g, rule = pair["by_delivered"][name]["gap_closed"], rc["by_delivered"][name]["rows"]["geometric_error"]["numbers"]["route_F1_road"]
            mf, zf = card["by_delivered"][name]["numbers"]["route_F1_road"], single_z["by_delivered"][name]["numbers"]["route_F1_road"]
            assert g["rule_row"] == "geometric_error" and abs(g["value"] - (mf - zf) / (rule - zf)) < 1e-9 and abs(g["rule_F1_road"] - rule) < 1e-12, (name, g)
        selfpair = __import__("json").load(open(E.score(args(["score", "--run", "m_s0", "--seed", "0", "--mode", "error", "--ref-run", "m_s0", "--ref-mode", "none", "--max", str(n), "--draws", "50"]), runs)))
        assert selfpair["ref"]["mode"] == "none" and os.path.exists(os.path.join(runs, "m_s0", f"route_card_error_vs_m_s0_rmnone_n{n}.json"))
        # a dump of another model.pt is refused
        torch.manual_seed(5)
        torch.save(dict(args=dict(seed=0, init_seed=0, cache=None, ident="e292_a", K=None, inputs="full", arm="z", steps=0), state=JR.RouteHead().state_dict()), os.path.join(runs, "z_s0", "model.pt"))
        try:
            rd("z_s0", "error")
            raise RuntimeError("a dump of another model.pt was read")
        except AssertionError:
            pass
        # a head is read only on the route cache it was trained on: another route cap / version in its args, or no record and a model older than the cache, is refused; the right record is read
        ident = E.route_cache_identity(side["ds"].rc.manifest)
        refused = lambda run_: E.dump(args(["dump", "--run", run_, "--seed", "0", "--mode", "error", "--cpu", "--max", "8"]), runs)  # noqa: E731
        for name_, rec in (("m_cap", dict(ident, max_samples=ident["max_samples"] + 1)), ("m_ver", dict(ident, version="v0")), ("m_ok", ident), ("m_train", dict(ident, tokens_sha="the TRAIN frames", split="train")), ("m_old", None)):
            torch.manual_seed(0)
            os.makedirs(os.path.join(runs, name_))
            torch.save(dict(args=dict(seed=0, init_seed=0, cache=None, ident="e292_a", K=None, inputs="full", arm="m", steps=0, **({} if rec is None else {"route_cache": rec})), state=JR.RouteHead().state_dict()), os.path.join(runs, name_, "model.pt"))
        os.utime(os.path.join(runs, "m_old", "model.pt"), (1e9, 1e9))
        for name_ in ("m_cap", "m_ver", "m_old"):
            try:
                refused(name_)
                raise RuntimeError(f"{name_}: a head of another route cache was dumped")
            except AssertionError:
                pass
        refused("m_ok")
        refused("m_train")                                                                       # the trainer records its TRAIN cache: the frame list differs from the VAL cache by design
    print(f"jroute_eval: real-cache dump / score / rules ok ({n} frames, route cache {ident['version']}, route cap {ident['max_samples']})")


if __name__ == "__main__":
    a = sys.argv[1:]
    test_selftest()
    test_cost()
    if a[:1] == ["only"]:
        test_mutants(a[1:])
    else:
        test_real_cache()
        if a[:1] != ["selftests"]:
            test_mutants()
