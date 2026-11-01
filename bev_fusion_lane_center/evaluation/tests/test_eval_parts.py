"""Spec test of evaluation/eval_parts.py (roadmap W11: route success, failure series, timing as parts of evaluate.py). CPU only, no GPU, no worker pool (about 90 s):
  * the route rule on a toy token item (known answers: a full route, half of it, a false branch) and the card keys;
  * STORED inputs reproduce STORED numbers through `eval_parts.run` (the code path of `evaluate.py --route --route-tokens ... --failure ... --series-dir ...`):
    route success of the champion with fork pruning from the stored tokens == drivers' result runs/e123_ladder/diag/pipeeval_f1_288_s0_fp0.3.json
    (success and route frames exactly, coverage / precision within the fp16 difference of the GPU run), the per-phase cards of the stored series graphs == the cards
    stored in the series file; the builder is never built on this path (build_net would raise);
  * timing: the command line of drivers/p18_latency.py for registry entries (the champion exactly; entries the driver cannot time are refused), the card keys of a stored
    latency json, the direction of every kind of key;
  * a card with parts refuses --subset.
The GPU paths (route from the builder, series run, p18_latency) are not run here: see evaluation/README.md for the verification commands.
Run (from the problem folder): wsl bash tools/env/_py14.sh test_eval_parts.py"""
import json
import os

import numpy as np

import eval_parts as EP
import evaluate as EV

R = EV.R
PIPE = f"{R}/runs/e123_ladder/diag/pipeeval_f1_288_s0_fp0.3.json"
LAT = f"{R}/runs/e18/latency/latency_M288_rec0_u2_fp0.3_e243_c_r1.json"
SERIES = "e130_champion"


def toy_item():
    """one piece (0,0) - (10,0) - (20,0) on the route, one piece (0,30) - (20,30) off it"""
    xy = np.array([[0, 0], [10, 0], [20, 0], [0, 30], [20, 30]], np.float32)
    return dict(xy=xy, pieces=[[0, 1, 2], [3, 4]]), np.stack([np.arange(21.0), np.zeros(21)], 1)


def test_route_rule_known_answers():
    it, rp = toy_item()
    full = EP.route_frame(np.array([.9, .9, .9, 0, 0]), it, rp)
    assert full["succ"] == 1 and full["cov"] == 1.0 and full["prec"] == 1.0
    half = EP.route_frame(np.array([.9, .9, 0, 0, 0]), it, rp)        # edge 1-2 has mean .45 < .5: the route is covered up to x = 10 + the 1.5 m tolerance (12 of 21 points)
    assert half["succ"] == 0 and abs(half["cov"] - 12 / 21) < 1e-9 and half["prec"] == 1.0
    edge = EP.route_frame(np.array([.5, .5, .5, 0, 0]), it, rp)       # mean exactly .5 counts as route
    assert edge["succ"] == 1
    false = EP.route_frame(np.array([.9, .9, .9, .9, .9]), it, rp)    # the false branch (21 points) next to the 22 points on the route
    assert false["succ"] == 0 and false["cov"] == 1.0 and abs(false["prec"] - 22 / 43) < 1e-9
    assert EP.route_frame(np.array([.9, .9, .9, 0, 0]), it, None) is None, "no route points, no result"
    m = EP.route_metrics([full, half, false])
    assert m["route_frames"] == 3 and abs(m["route_success"] - 1 / 3) < 1e-12 and abs(m["route_cov"] - (1 + 12 / 21 + 1) / 3) < 1e-12
    assert np.isnan(EP.route_metrics([])["route_success"]) and EP.route_metrics([])["route_frames"] == 0


def test_direction_of_every_kind_of_key():
    d = EP.direction(["route_success", "route_cov", "route_frames", "ms_per_frame", "ms.total.p90", "series.dark_single.fail.Q", "series.dark_single.fail.vert100",
                      "series.dark_single.cl.route_success", "series.dark_maxdecay_0.95.fail.false_len_frac", "series.dark_maxdecay_0.95.fail.F1@1.0"])
    assert d.pop("series.dark_maxdecay_0.95.fail.false_len_frac") == "lower" and d.pop("series.dark_maxdecay_0.95.fail.F1@1.0") == "higher", "a dot in the tag or the metric"
    assert d == {"route_success": "higher", "route_cov": "higher", "ms_per_frame": "lower", "ms.total.p90": "lower", "series.dark_single.fail.Q": "higher",
                 "series.dark_single.fail.vert100": "lower", "series.dark_single.cl.route_success": "higher"}, d


def test_timing_command_and_keys():
    reg = EV.registry()
    ch = reg["champion"]
    assert EP.timing_argv(ch, reg, 60, "_t") == ["p18_latency.py", "--name", "e123_D7_r3_f1", "--M", "288", "--K0", "144", "--n", "60", "--rec", "0", "--union", "2",
                                                  "--fork_prune", "0.3", "--tag", "_t"]
    other = dict(ch, asm=dict(ch["asm"], cov_r=0.6))
    for bad in (other, dict(ch, tau=0.5), dict(ch, design="D6"), dict(ch, set=dict(ch["set"], extra=1)), dict(reg["control"])):
        try:
            EP.timing_argv(bad, reg)
        except (ValueError, KeyError):
            continue
        raise AssertionError("an entry the driver cannot time was accepted")
    for name, e in reg.items():  # every entry is either timed as it is or refused with a ValueError, never a half-right command
        try:
            a = EP.timing_argv(e, reg)
        except (ValueError, KeyError):
            continue
        assert a[a.index("--name") + 1] and (("--name2" in a) == (e["kind"] == "union")), name
    if os.path.exists(LAT):
        res = json.load(open(LAT))
        m = EP.timing_metrics(res)
        assert m["ms_per_frame"] == res["total"]["median_ms"] == m["ms.total.median"] and m["ms.builder.p90"] == res["builder"]["p90_ms"] and "ms.memory.median" in m


def test_stored_inputs_reproduce_stored_numbers_through_run():
    z = json.load(open(PIPE)) if os.path.exists(PIPE) else None
    sf = f"{R}/runs/e17/{SERIES}/dark_single.pkl"
    if z is None or not os.path.exists(sf) or not os.path.exists(os.path.expanduser("~/exp_gcar_b123/p10/tok_p18_f1pr_val_clean.pkl")):
        print("SKIP stored-input test: stored results or inputs are missing in this checkout")
        return

    def no_gpu(e):
        raise AssertionError("the builder was built on a stored-input path")

    reg = EV.registry()
    out = EP.run(reg["champion"], reg, "val", dict(route=dict(head=EP.ROUTE_HEAD, tokens="tok_p18_f1pr_val_clean", step=4.0),
                                                   failure=dict(perts=["dark"], variant="single", series_dir=SERIES, suffix="", bs=4, workers=0)), no_gpu)
    m = out["metrics"]
    assert m["route_frames"] == z["route_frames"] == 823 and abs(m["route_success"] - z["route_success"]) < 1e-9, (m["route_success"], z["route_success"])
    assert abs(m["route_cov"] - z["route_cov"]) < 3e-3 and abs(m["route_prec"] - z["route_prec"]) < 3e-3
    import pickle
    st = pickle.load(open(sf, "rb"))["cards"]
    for ph in ("cl", "fail"):
        for k, ks in (("S", "S"), ("F1@1.0", "F1@1.0"), ("J", "J"), ("vert100", "vert100"), ("lanes_found", "lanes_found")):
            assert abs(m[f"series.dark_single.{ph}.{k}"] - st[ph][ks]) < 1e-9, (ph, k)
    assert not any(k.endswith(".route_success") and k.startswith("series.") for k in m), "stored graphs carry no route: no GPU route per phase"
    assert out["info"]["series"]["dark_single"]["n"] == {"cl": 137, "fail": 154} and set(out["better"]) == set(m) - {"route_frames"}


def test_parts_refuse_subset_and_fresh():
    reg = EV.registry()
    for kw in (dict(subset="val250"), dict(fresh=True)):
        try:
            EV.evaluate("champion", reg, "val", parts=dict(timing=dict(n=1, tag="")), **kw)
        except ValueError:
            continue
        raise AssertionError("parts ran on a subset / fresh card")


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_eval_parts: {n} tests passed")
