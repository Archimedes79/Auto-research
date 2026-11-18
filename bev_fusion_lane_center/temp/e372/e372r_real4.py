"""throw-away (E372 live read-out, not a verdict): on the first 4 VAL frames, CPU, in a temporary runs folder under temp/e372 (nothing under runs/ is written):
(1) the cache path of the OLD module (evaluation/jtoken_eval.py) and of the scratch copy give the same dump bytes and the same cards (but date / commit) for the champion head e373_cb_s0;
(2) the scratch copy's `dump --live` with the real B' runs/e292_a_s0 on the CPU, `score --live --ref-run e373_cb_s0 --ref-cache` at the D115 cell, `live-diff` of the live dump against a copy of itself"""
import importlib.util
import os
import pickle
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ["BFLC_ROOT"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NEW = load("jtoken_eval", os.path.join(HERE, "e372r_jtoken_eval.py"))
OLD = load("jtoken_eval_old", os.path.join(ROOT, "evaluation", "jtoken_eval.py"))
import torch  # noqa: E402

torch.set_num_threads(2)
RUN, N = "e373_cb_s0", 4
D115 = ["--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate"]
with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    runs = {k: os.path.join(tmp, k, "runs") for k in ("old", "new")}
    outs = {k: os.path.join(tmp, k, "out") for k in runs}
    for k in runs:
        os.makedirs(os.path.join(runs[k], RUN))
        os.makedirs(outs[k])
        shutil.copy(os.path.join(ROOT, "runs", RUN, "model.pt"), os.path.join(runs[k], RUN, "model.pt"))
    cards = {}
    for k, E in (("old", OLD), ("new", NEW)):
        E.dump(E.parse_args(["dump", "--run", RUN, "--seed", "0", "--cpu", "--max", str(N)]), runs[k])
        cards[k] = E.score(E.parse_args(["score", "--run", RUN, "--seed", "0", "--max", str(N)] + D115), runs[k], outs[k])["rec"]
    b = {k: open(os.path.join(runs[k], RUN, f"token_val_n{N}.pkl"), "rb").read() for k in runs}
    strip = lambda c: {k: v for k, v in c.items() if k not in ("date", "commit")}  # noqa: E731
    print(f"(1) cache path old vs new: dump bytes identical {b['old'] == b['new']} ({len(b['new'])} bytes); D115 card identical but date / commit {strip(cards['old']) == strip(cards['new'])}; topo F1 {cards['new']['metrics']['topo_F1']:.4f}", flush=True)
    E, rn, out = NEW, runs["new"], outs["new"]
    t0 = time.time()
    E.dump(E.parse_args(["dump", "--live", "--run", RUN, "--seed", "0", "--cpu", "--max", str(N)]), rn)
    t_live = time.time() - t0
    d = pickle.load(open(os.path.join(rn, RUN, f"token_val_live_n{N}.pkl"), "rb"))
    dc = pickle.load(open(os.path.join(rn, RUN, f"token_val_n{N}.pkl"), "rb"))
    m = d["meta"]
    print(f"(2) live dump ({t_live:.0f} s for {N} frames on the CPU): b_source kind {m['b_source']['kind']}, sha_nodes {m['b_source']['sha_nodes'][:16]}, bs_b {m['bs_b']} bs_head {m['bs_head']}, tokens_exact {m['tokens_exact']} "
          f"({m['tokens_exact_frames']} of {N} frames), {os.path.getsize(os.path.join(rn, RUN, f'token_val_live_n{N}.pkl')) / N / 1e3:.0f} kB per frame (cache dump {os.path.getsize(os.path.join(rn, RUN, f'token_val_n{N}.pkl')) / N / 1e3:.0f} kB)", flush=True)
    for f, g in zip(dc["frames"], d["frames"]):
        common = len(set(f["cell"].tolist()) & set(g["cell"].tolist()))
        print(f"   frame {f['token'][:8]}: same cell list {bool((f['cell'] == g['cell']).all())}, cells in both {common} of {len(f['cell'])}, kept nodes cache {int(f['keep'].sum())} live {int(g['keep'].sum())}, record links cache {len(f['rec']['src'])} live {len(g['rec']['src'])}")
    par = E.score(E.parse_args(["score", "--live", "--run", RUN, "--seed", "0", "--max", str(N), "--ref-run", RUN, "--ref-cache"] + D115), rn, out)["rec"]
    pr = par["info"]["paired_vs_ref_run"]
    print(f"   parity read (D115, {N} frames, CPU live vs GPU cache): topo F1 live {par['metrics']['topo_F1']:.4f} cache {par['info']['ref_run']['topo_F1']:.4f} diff {pr['topo_F1']['diff']:+.4f}; junction F1 diff {pr['junction_F1']['diff']:+.4f}; "
          f"card {sorted(os.listdir(out))}", flush=True)
    os.makedirs(os.path.join(rn, "copy"))
    shutil.copy(os.path.join(rn, RUN, "model.pt"), os.path.join(rn, "copy", "model.pt"))
    d["meta"]["run"] = "copy"
    pickle.dump(d, open(os.path.join(rn, "copy", f"token_val_live_n{N}.pkl"), "wb"))
    E.live_diff(E.parse_args(["live-diff", "--run", "copy", "--against", RUN, "--seed", "0", "--max", str(N)]), rn, out)
print("real4 done")
