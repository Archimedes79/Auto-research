"""Cost of the plain ENSEMBLE of two token heads (E379 arm e2; the D106 gate of the card: is two heads at inference cheap enough to be the simpler champion than one distilled head?). The protocol of `jtoken_eval.py timing` (batch 1, bf16 autocast on the GPU, CUDA events, the first
`--frames` cached VAL frames, `--repeats` passes after one untimed pass, the host / GPU quiet verdict of `jtiming.py`), for the head alone, two heads one after the other, and two heads STACKED in one forward (`torch.func.stack_module_state` + `vmap` over `functional_call`: the head is launch-bound,
a stacked forward should cost far less than twice a head), each with the probabilities and their mean (what an ensemble decode reads: `jtoken.probs` of both, the mean of p and of the type probabilities), alone and, with --b-run, behind B' (the whole path).

    wsl bash tools/env/_pyr.sh token_time_ens.py --runs e373_cb_s0,e367_b25_s0 --b-run runs/e292_a_s0 [--frames 24] [--repeats 5] [--cpu]     # GPU only when `jtoken_eval.gpu_free` says so; writes runs/A/token_timing_ens_<stamp>.json
    wsl bash tools/env/_py14.sh token_time_ens.py selftest                                                                                     # CPU, no data: the stacked forward equals the two single forwards, the mean of the probabilities by hand

Result keys: params (one head, two heads), median_ms / p90_ms of head, head_probs, seq2_probs, stacked_probs (and bprime, whole_1, whole_seq2, whole_stacked with --b-run), `stacked_error` when the head is not vmappable (the other numbers are still written), `quiet` / `usable_for_gcost` (true only for a quiet host and an idle GPU),
`stacked_max_diff` (the largest difference of p between the stacked and the sequential forward, a check, not a metric). Not measured here: a CUDA-graph forward of the ensemble (jgraph.py's route), another batch size.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training/data", "training/grid", "training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtoken as JK  # noqa: E402


def mean_probs(pr_list):
    """the ensemble probabilities of a list of `jtoken.probs` dicts: the mean of p (link probability) and of pt (the 7-way type probabilities) -> (p, pt)"""
    return sum(p["p"] for p in pr_list) / len(pr_list), sum(p["pt"] for p in pr_list) / len(pr_list)


def stacked_forward(base, params, buffers, batch):
    """both heads in one forward: `vmap` over the stacked parameters of `functional_call(base, ...)` on the SAME batch -> the output dict with a leading head axis"""
    f = lambda p_, b_, x_: torch.func.functional_call(base, (p_, b_), (x_,))  # noqa: E731
    return torch.vmap(f, in_dims=(0, 0, None), randomness="same")(params, buffers, batch)


def probs_of_stacked(out, n=2):
    """the `jtoken.probs` dicts of the heads of a stacked output (the leading axis sliced)"""
    return [JK.probs({k: (v[i] if torch.is_tensor(v) else v) for k, v in out.items()}) for i in range(n)]


def check_stacked(m1, m2, batch):
    """the stacked forward gives the two single forwards: the largest |p| difference (float noise only)"""
    params, buffers = torch.func.stack_module_state([m1, m2])
    base = copy.deepcopy(m1).to("meta")
    out = stacked_forward(base, params, buffers, batch)
    with torch.no_grad():
        want = [JK.probs(m(batch)) for m in (m1, m2)]
    got = probs_of_stacked(out)
    return max(float((g["p"] - w["p"]).abs().max()) for g, w in zip(got, want)), (params, buffers, base)


def selftest():
    torch.manual_seed(0)
    b = JK.synthetic_batch(1, 48, 0)
    ms = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        ms.append(JK.JTokenHead(d=32, heads=4, layers=2, ffn=64).eval())
    diff, _ = check_stacked(ms[0], ms[1], b)
    assert diff < 1e-5, f"the stacked forward is not the two single forwards: {diff}"
    pr = [JK.probs(m(b)) for m in ms]
    p, pt = mean_probs(pr)
    assert torch.allclose(p, (pr[0]["p"] + pr[1]["p"]) / 2) and torch.allclose(pt, (pr[0]["pt"] + pr[1]["pt"]) / 2) and torch.allclose(pt.sum(-1), torch.ones_like(pt.sum(-1)), atol=1e-5), "the mean of the probabilities"
    assert float((pr[0]["pt"] - pr[1]["pt"]).abs().max()) > 1e-4, "the two heads differ in their type probabilities too"
    assert float((pr[0]["p"] - pr[1]["p"]).abs().max()) > 1e-3, "the two heads differ (a test of the test)"
    p1, _ = mean_probs([pr[0]])
    assert torch.equal(p1, pr[0]["p"]), "the mean of one head is the head"
    print("token_time_ens selftest ok")


def main():
    import jrecur_time as JRT
    import jtiming as JTG
    import jtoken_eval as JE
    import jtoken_train as JKT
    import lane_ptok_data as LP
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="A,B: two runs of the same architecture and cache")
    ap.add_argument("--b-run", default=None, help="a QGraph run (runs/e292_a_s<S>): also B' alone and the whole paths")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    ra, rb = a.runs.split(",")
    m1, args1 = JKT.load_model(ra, "cpu", JE.RUNS)
    m2, args2 = JKT.load_model(rb, "cpu", JE.RUNS)
    kw1, kw2 = JE.cache_kw(args1), JE.cache_kw(args2)
    assert (kw1["seed"], kw1["ident"], kw1["K"]) == (kw2["seed"], kw2["ident"], kw2["K"]), "the two heads are not on the same cache"
    ok, why = (False, "--cpu") if a.cpu else JE.gpu_free()
    if not a.cpu and not ok:
        sys.exit(f"STOP: no GPU timing ({why}); --cpu times the code path on the CPU only")
    dev = "cpu" if a.cpu else "cuda"
    ds = JE.open_cache("val", kw1["seed"], kw1["ident"], kw1["K"], kw1.get("cache"))
    K = ds.K
    m1, m2 = m1.to(dev).eval(), m2.to(dev).eval()
    cut = lambda b: {k: v[:, :K] for k, v in b.items() if k in JK.INPUTS}  # noqa: E731
    items = [JKT.to_device(cut(LP.collate([ds[i]])), dev) for i in range(a.frames)]
    ac = lambda: torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")  # noqa: E731
    cuda = dev == "cuda"
    stacked_error, stacked_max_diff, st = None, None, None
    try:
        with torch.no_grad():
            stacked_max_diff, st = check_stacked(m1, m2, items[0])
    except Exception as e:                                                                  # not vmappable: said, the rest is still measured
        stacked_error = f"{type(e).__name__}: {str(e)[:300]}"
    mb = xs = None
    if a.b_run:
        import lane_tok_data as TD
        import qpredict as QP
        mb = QP.load_model(JE._abs(a.b_run), dev)
        td = TD.TokData("val", JE.LATTICE, feats=True)
        xs = [td[i]["x"][None].to(dev) for i in range(a.frames)]
    one = lambda b: JK.probs(m1(b))  # noqa: E731
    seq2 = lambda b: mean_probs([JK.probs(m1(b)), JK.probs(m2(b))])  # noqa: E731
    stk = (lambda b: mean_probs(probs_of_stacked(stacked_forward(st[2], st[0], st[1], b)))) if st is not None else None  # noqa: E731
    load0, gpu0 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    names = dict(head=lambda b: m1(b), head_probs=one, seq2_probs=seq2)
    if stk:
        names["stacked_probs"] = stk
    if mb is not None:
        names["bprime"] = None
        for k, f in (("whole_1", one), ("whole_seq2", seq2)) + ((("whole_stacked", stk),) if stk else ()):
            names[k] = f
    ms = {k: [] for k in names}
    for rep in range(a.repeats + 1):
        row = {k: [] for k in names}
        for i, b in enumerate(items):
            with torch.no_grad(), ac():
                for k, f in names.items():
                    if k == "bprime":
                        row[k].append(JRT.timed(lambda: JE.live_tokens(mb, xs[i], K), cuda)[1])
                    elif k.startswith("whole_"):
                        row[k].append(JRT.timed(lambda: f(JE.live_tokens(mb, xs[i], K)), cuda)[1])
                    else:
                        row[k].append(JRT.timed(lambda: f(b), cuda)[1])
        if rep:                                                                              # the first pass is untimed
            for k in ms:
                ms[k].append(row[k])
    load1, gpu1 = JTG.load1(), (None if a.cpu else JTG.gpu_state())
    quiet, line = JTG.quiet_verdict(load0, load1, dict(util_max=0.0) if a.cpu else gpu0, dict(util_max=0.0) if a.cpu else gpu1)
    res = dict(runs=[ra, rb], device=dev, K=K, frames=a.frames, repeats=a.repeats, params=dict(head=JK.count_params(m1), two_heads=JK.count_params(m1) + JK.count_params(m2)), host=line, quiet=bool(quiet), usable_for_gcost=bool(quiet and cuda), gpu_rule=why, bf16=cuda,
               median_ms={k: float(np.median(v)) for k, v in ms.items()}, p90_ms={k: float(np.percentile(v, 90)) for k, v in ms.items()}, stacked_max_diff=stacked_max_diff, stacked_error=stacked_error)
    if mb is not None:
        res["params_bprime"] = JK.count_params(mb.pyr) + JK.count_params(mb.nodes)
        res["params_whole_path"] = dict(one_head=res["params_bprime"] + res["params"]["head"], two_heads=res["params_bprime"] + res["params"]["two_heads"])
    print(json.dumps(res, indent=1))
    print(f"written {os.path.relpath(JE._write_stamped(os.path.join(JE.RUNS, ra), 'token_timing_ens' + ('_cpu' if a.cpu else ''), res), ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
