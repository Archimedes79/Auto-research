"""Spec test of evaluation/eval_parts.py (roadmap W11: route success, failure series, timing as parts of evaluate.py). CPU only, no GPU, no worker pool (about 90 s):
  * the route rule on a toy token item (known answers: a full route, half of it, a false branch) and the card keys;
  * STORED inputs reproduce STORED numbers through `eval_parts.run` (the code path of `evaluate.py --route --route-tokens ... --failure ... --series-dir ...`):
    route success of the champion with fork pruning from the stored tokens == drivers' result runs/e123_ladder/diag/pipeeval_f1_288_s0_fp0.3.json
    (success and route frames exactly, coverage / precision within the fp16 difference of the GPU run), the per-phase cards of the stored series graphs == the cards
    stored in the series file; the builder is never built on this path (build_net would raise);
  * a frame with route points and an EMPTY graph is a failure (route_over, phase_route), not skipped; the data part of --split test is 'val' (the TEST rows live in the VAL pool);
  * timing: the command line of drivers/p18_latency.py for registry entries (the champion exactly; entries the driver cannot time are refused, also a union whose members
    differ in M / K0), a stale latency file is refused, the card keys of a stored latency json, the direction of every kind of key;
  * a card with parts refuses --subset.
The GPU paths (route from the builder, series run, p18_latency) are not run here: see evaluation/README.md for the verification commands.
Run (from the problem folder): wsl bash tools/env/_py14.sh test_eval_parts.py"""
import contextlib
import json
import os
import sys
import tempfile
import time
import types

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


def test_empty_graph_is_a_failure_not_a_skip():
    import p8_lib as L
    import sd_lib as SL
    it, rp = toy_item()
    rows = {0: (rp, it), 1: (rp, None), 2: (rp, dict(xy=np.zeros((0, 2), np.float32))), 3: (None, it), 4: (rp, it)}  # 1, 2: route points but no graph; 3: no route; 4: no SD map
    sdl = {0: 1, 1: 1, 2: 1, 3: 1, 4: None}
    old = (L.cache, L.draw_sd, SL.raster_sd)
    L.cache = lambda name, part: {"sd": sdl, "rpts": {k: v[0] for k, v in rows.items()}}[name]
    L.draw_sd = lambda *a: (None, None, None)
    SL.raster_sd = lambda m: np.zeros((1, 4, 4), np.float32)
    try:
        res, n = EP.route_over(((k, v[1]) for k, v in rows.items()), lambda i, r: np.array([.9, .9, .9, 0, 0]), "val")
    finally:
        L.cache, L.draw_sd, SL.raster_sd = old
    m = EP.route_metrics(res)
    assert n == 5 and m["route_frames"] == 3 and m["route_empty"] == 2 and abs(m["route_success"] - 1 / 3) < 1e-12 and abs(m["route_cov"] - 1 / 3) < 1e-12, (n, m)
    meta = [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (0, 1)]  # (row, phase) of a series; run_series gives None for a skip: empty graph (row 1), no route (3), no SD map (4)
    v = EP.phase_route([1.0, None, None, None, None, 0.0], meta, {k: v[0] for k, v in rows.items()}, sdl)
    assert v["cl"] == (0.5, 2, 1) and v["fail"] == (0.0, 2, 1), v  # cl: rows 0, 1 -> (1 + 0) / 2; fail: rows 2 (empty), 0 (0.0) -> 0 / 2


def test_route_from_builder_reads_the_data_part_of_the_split():
    """--split test --route must read sd_val.pkl / route_pts_val.pkl: the TEST rows are in the VAL pool (BuilderData.part), there is no *_test file"""
    import p16_pipeline as PP
    got, old = [], (EP.load_head, EP.builder_items, EP.route_over, PP.FastRoute)
    EP.load_head, EP.builder_items, PP.FastRoute = (lambda h, d: None), (lambda *a, **k: iter(())), (lambda m: None)
    EP.route_over = lambda items, probs, part: (got.append(part), ([], 0))[1]
    try:
        for split in ("val", "test"):
            EP.route_from_builder(None, types.SimpleNamespace(temporal=False), None, split)
    finally:
        EP.load_head, EP.builder_items, EP.route_over, PP.FastRoute = old
    assert got == ["val", "val"], got


@contextlib.contextmanager
def fake_modules(**fakes):
    """modules that the code under test imports inside a function, replaced for the block (the heavy ones need a GPU)"""
    old = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        yield
    finally:
        for k, v in old.items():
            sys.modules.pop(k) if v is None else sys.modules.__setitem__(k, v)


def test_route_builder_drops_dp_eps_and_keeps_the_rest_of_the_assembly():
    """the `*_dp` registry entries carry asm dp_eps; the route input needs aux=True, which assemble refuses together with dp_eps > 0: the route path assembles without it"""
    calls = []
    ns = lambda **kw: types.SimpleNamespace(**kw)                  # noqa: E731

    class DS:
        rows = [5]

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return {}

    with fake_modules(p16_data=ns(collate=lambda xs: {"k": [0]}), p16_timing=ns(to_gpu=lambda b: b),
                      p16_infer=ns(assemble=lambda r, b, tau, cfg, **kw: (calls.append(kw), ("g", {"V": [1]}))[1]),
                      p16_pipeline=ns(route_input=lambda *a: "item"), p16_runtime=ns(FastBuilder=lambda net, cfg, b: ns(raw=lambda b: {"pi": 0}, h_static=0))):
        items = list(EP.builder_items(None, ns(temporal=False), dict(tau=0.3, asm=dict(cov_r=1.0, dp_eps=0.2, fast=2)), DS()))
    assert items == [(5, "item")] and calls == [dict(aux=True, cov_r=1.0, fast=2)], (items, calls)


def test_route_series_drops_dp_eps_too():
    """the failure series route path (p17_route.run_series assembles with aux=True) gets the same assembly settings without dp_eps"""
    got = []
    ns = lambda **kw: types.SimpleNamespace(**kw)                  # noqa: E731
    old_head, EP.load_head = EP.load_head, lambda h, d: None
    try:
        with fake_modules(p17_eval=ns(SeriesData=lambda tag, with_prev: ns()), p8_lib=ns(cache=lambda name, part: {}),
                          p17_route=ns(run_series=lambda net, cfg, ds, rnets, sigmas, tau, asm: (got.append(asm), ({(0, 0): []}, [], None))[1])):
            EP.series_route(None, ns(temporal=False), dict(tau=0.3, asm=dict(cov_r=1.0, dp_eps=0.2)), "dark_single")
    finally:
        EP.load_head = old_head
    assert got == [dict(cov_r=1.0)], got


def test_compare_across_the_dp_basis_warns_that_s_is_not_comparable():
    """S of a `*_dp` entry gains from the lower node-id count (D25: recall -> recall_dp +0.0115 in S, Q -0.0005): compare shows a warning, the card says it"""
    import contextlib
    import io
    assert EV.s_comparable("recall", "champion") and EV.s_comparable("recall_dp", "champion_dp") and not EV.s_comparable("recall_dp", "recall") and not EV.s_comparable("recall", "recall_dp")
    card = dict(name="recall_dp", split="val", subset="", fresh=False, frames=1, commit="c", info=dict(asm_ms=0.0, metric_s=0.0), metrics=dict(S=0.7789), path="p",
                compare=dict(name="recall", metrics=dict(S=0.7675), paired=dict(S=dict(diff=0.0115, lo=0.009, hi=0.014, verdict="better")), s_not_comparable=True))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        EV.show(card)
        EV.show(dict(card, compare=dict(card["compare"], s_not_comparable=False)))
    first, second = out.getvalue().split("card: p")[:2]
    assert "WARNING: S is NOT comparable between recall_dp and recall" in first and "Read Q" in first and "WARNING" not in second


def test_direction_of_every_kind_of_key():
    d = EP.direction(["route_success", "route_cov", "route_frames", "route_empty", "ms_per_frame", "ms.total.p90", "series.dark_single.fail.Q", "series.dark_single.fail.vert100",
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
    union = dict(kind="union", members=["a", "b"], asm=ch["asm"], tau=ch["tau"])  # p18_latency times the second member at the first one's M / K0
    reg2 = dict(reg, a=ch, b=dict(ch, run="other", set=dict(ch["set"])), c=dict(ch, run="other", set=dict(ch["set"], K0=100)))
    assert EP.timing_argv(union, reg2)[-2:] == ["--name2", "other"]
    try:
        EP.timing_argv(dict(union, members=["a", "c"]), reg2)
    except ValueError:
        pass
    else:
        raise AssertionError("a union whose members differ in K0 was timed at the first member's K0")
    if os.path.exists(LAT):
        res = json.load(open(LAT))
        m = EP.timing_metrics(res)
        assert m["ms_per_frame"] == res["total"]["median_ms"] == m["ms.total.median"] and m["ms.builder.p90"] == res["builder"]["p90_ms"] and "ms.memory.median" in m


def test_timing_run_refuses_a_stale_latency_file():
    reg = EV.registry()
    with tempfile.TemporaryDirectory() as tmp:
        d = f"{tmp}/runs/e18/latency"
        os.makedirs(d)
        stale = f"{d}/latency_old_t.json"
        json.dump({"total": {"median_ms": 99.0}}, open(stale, "w"))
        os.utime(stale, (time.time() - 100, time.time() - 100))
        calls = []
        fake = types.SimpleNamespace(main=lambda: calls.append(1))  # writes nothing
        old = (EP.R, sys.modules.get("p18_latency"))
        EP.R, sys.modules["p18_latency"] = tmp, fake
        try:
            try:
                EP.timing_run(reg["champion"], reg, 1, "_t")
            except RuntimeError:
                pass
            else:
                raise AssertionError("the stale file of an earlier run was returned")
            def write_new():
                with open(f"{d}/latency_new_t.json", "w") as fh:
                    json.dump({"total": {"median_ms": 5.0}}, fh)
            fake.main = write_new
            res, f = EP.timing_run(reg["champion"], reg, 1, "_t")
            assert res["total"]["median_ms"] == 5.0 and f.endswith("latency_new_t.json"), (res, f)
            def rewrite_old():                                       # the old file written again (other size, a mtime that no clock of this machine has to agree with)
                with open(stale, "w") as fh:
                    json.dump({"total": {"median_ms": 7.0}, "pad": "x" * 20}, fh)
                os.utime(stale, (time.time() - 50, time.time() - 50))
            os.remove(f"{d}/latency_new_t.json")
            fake.main = rewrite_old
            res, f = EP.timing_run(reg["champion"], reg, 1, "_t")
            assert res["total"]["median_ms"] == 7.0 and f == stale, (res, f)
        finally:
            EP.R = old[0]
            if old[1] is None:
                sys.modules.pop("p18_latency")
            else:
                sys.modules["p18_latency"] = old[1]


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
    assert m["route_empty"] == 0 and abs(m["route_success"] - 0.8991) < 5e-5, m  # no frame is empty there: counting empty graphs as failures leaves the stored number as it was
    assert abs(m["route_cov"] - z["route_cov"]) < 3e-3 and abs(m["route_prec"] - z["route_prec"]) < 3e-3
    import pickle
    st = pickle.load(open(sf, "rb"))["cards"]
    for ph in ("cl", "fail"):
        for k, ks in (("S", "S"), ("F1@1.0", "F1@1.0"), ("J", "J"), ("vert100", "vert100"), ("lanes_found", "lanes_found")):
            assert abs(m[f"series.dark_single.{ph}.{k}"] - st[ph][ks]) < 1e-9, (ph, k)
    assert not any(k.endswith(".route_success") and k.startswith("series.") for k in m), "stored graphs carry no route: no GPU route per phase"
    assert out["info"]["series"]["dark_single"]["n"] == {"cl": 137, "fail": 154} and set(out["better"]) == set(m) - {"route_frames", "route_empty"}


def test_parts_refuse_subset_and_fresh():
    reg = EV.registry()
    for kw in (dict(subset="val250"), dict(fresh=True)):
        try:
            EV.evaluate("champion", reg, "val", parts=dict(timing=dict(n=1, tag="")), **kw)
        except ValueError:
            continue
        raise AssertionError("parts ran on a subset / fresh card")


def test_a_stored_input_of_the_test_split_needs_split_test():
    """review 2026-11-10: `--split val --route-tokens tok_..._test_clean` scored TEST frames without a row of the TEST log; a stored input that names the TEST split is refused
    under any other split (the name is the best effort; `latest` and `val` names pass)"""
    assert EP.names_test("tok_p18_f1pr_test_clean") and EP.names_test("Test") and EP.names_test("e130_test.series") and EP.names_test("x-test")
    assert not EP.names_test("tok_p18_f1pr_val_clean") and not EP.names_test("latest") and not EP.names_test("contest_val") and not EP.names_test("")
    for parts in ({"route": {"tokens": "tok_p18_f1pr_test_clean", "head": "h", "step": 1}}, {"failure": {"series_dir": "e130_test", "perts": [], "variant": "v"}},
                  {"failure": {"series_dir": "e130", "suffix": "_test", "perts": [], "variant": "v"}}):
        try:
            EP.run({}, {}, "val", parts, None)
        except ValueError as e:
            assert "TEST" in str(e), e
            continue
        raise AssertionError(f"a stored TEST input was read under --split val: {parts}")
    raised = None
    try:        # under --split test the name is no reason to refuse (evaluate.test_allowed gates the split itself): it goes on and fails only because the file is not there
        EP.run({}, {}, "test", {"route": {"tokens": "tok_no_such_test_clean", "head": "h", "step": 1}}, None)
    except Exception as e:  # noqa: BLE001
        raised = e
    assert raised is not None and "names the TEST split" not in str(raised), raised


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"test_eval_parts: {n} tests passed")
