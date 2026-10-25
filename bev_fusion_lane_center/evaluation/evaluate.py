"""The one evaluation entry point: a named model (models/registry.json) on a split, with its standard settings, all metrics, one card.

    tools/env/_py14.sh evaluate.py champion                       # VAL (873 frames), writes runs/eval/champion/val.json
    tools/env/_py14.sh evaluate.py fast --compare champion        # + paired scene bootstrap against another name (same frames)
    tools/env/_py14.sh evaluate.py champion --asm fork_prune=0.4  # try a setting (recorded in the card as an override)
    tools/env/_py14.sh evaluate.py champion --split test --test-id E230.f1p   # TEST only when the id is logged in evaluation/TEST_USAGE_LOG.md
    tools/env/_py14.sh evaluate.py --list

What it does
  * builder: the raw network outputs of (run, split, M, K0) are computed once on the GPU and cached in ~/exp_gcar_b123/eval_cache/ (the
    loop-18 caches ~/exp_gcar_b123/p18/raw_val873_<run>_<M>.pkl are reused); assembly settings and thresholds are re-scored on the CPU in seconds
  * graphs: stored graphs of a system that is not re-run (the control E94)
  * metrics: sparse_metrics.card_any(extra=True) - S, Q, F1, J, FL, vertices per 100 m, lanes found, topo P / R / F1, dup_frac, recall by range
  * card: runs/eval/<name>/<split>[_<tag>].json with the registry entry, overrides, frames, cache file, code commit, metrics, assembly ms per frame
Timing of the network (ms per frame on an idle GPU) is not measured here: evaluation/drivers/p17_latency.py (protocol in PROTOCOL.md).
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

R = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
HOME = os.path.expanduser("~")
CACHE = f"{HOME}/exp_gcar_b123/eval_cache"
SPLITS = {"val": ("val_clean", "val"), "test": ("test_clean", "test")}
SHOW = ("S", "Q", "F1@1.0", "J", "topo_F1", "lanes_found", "R@1_far", "false_len_frac", "dup_frac", "vert100")


def registry():
    return {k: v for k, v in json.load(open(f"{R}/models/registry.json", encoding="utf-8")).items() if not k.startswith("_")}


def commit():
    try:
        h = subprocess.run(["git", "-C", R, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", R, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10).stdout.strip()
        return (h + ("+dirty" if dirty else "")) if h else "no-git"
    except Exception:
        return "no-git"


def test_allowed(test_id):
    log = open(f"{R}/evaluation/TEST_USAGE_LOG.md", encoding="utf-8").read()
    return bool(test_id) and test_id in log


def builder_raw(e, split):
    """(raw node outputs, ground truths) of every frame of the split; cached"""
    run, M, K0 = e["run"], e["set"]["M"], e["set"]["K0"]
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/raw_{split}_{run}_{M}_{K0}.pkl"
    old = [f"{HOME}/exp_gcar_b123/p18/raw_val873_{run}_{M}.pkl"] + ([f"{HOME}/exp_gcar_b123/p18/raw_val873.pkl"] if run == "e123_D7_r3_f1" and M == 288 else [])
    for p in [path] + (old if split == "val" else []):
        if os.path.exists(p):
            raw, gts = pickle.load(open(p, "rb"))
            return raw, gts, p
    import torch
    import p16_data as PD
    import p16_infer as PI
    import p16_train as PT
    c = json.load(open(f"{R}/runs/e123_ladder/cards/{run}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
    net, cfg = PT.build(e["design"], extra)
    net = net.cuda()
    net.load_state_dict(torch.load(f"{R}/runs/e123/{run}/model.pt", map_location="cuda"), strict=False)
    ds = PD.BuilderData(SPLITS[split][0], ks=None, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, list(range(len(ds))), taus=(e["tau"],), bs=8, workers=2, raw=raw)
    pickle.dump((raw, gts), open(path, "wb"))
    return raw, gts, path


def graphs_of(name, e, split, asm_over=None, tau=None):
    """graphs, ground truths, info for a registry entry"""
    if e["kind"] == "graphs":
        z = pickle.load(open(f"{R}/{e['files'][split]}", "rb"))
        import p16_data as PD
        ds = PD.BuilderData(SPLITS[split][0], ks=None, train=False)
        gts = [ds.gt[ds.rows[int(k)]] for k in z["ks"]]
        return z["preds"], gts, dict(source=e["files"][split], asm_ms=float("nan"))
    import p16_infer as PI
    import p16_train as PT
    raw, gts, cache = builder_raw(e, split)
    c = json.load(open(f"{R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
    import p16_builder as PB
    kw = dict(PT.DESIGNS[e["design"]])
    kw.update(extra)
    cfg = PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)})  # the config of PT.build without building the network
    asm = dict(e["asm"])
    asm.update(asm_over or {})
    t = e["tau"] if tau is None else tau
    t0 = time.time()
    g = [PI.assemble(r, None, t, cfg, **asm) for r in raw]
    return g, gts, dict(cache=cache, asm=asm, tau=t, asm_ms=(time.time() - t0) / max(len(raw), 1) * 1e3)


def scenes(split):
    sp = json.load(open(f"{R}/data/splits.json"))
    return np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["scene"][np.array(sp[SPLITS[split][1]])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--split", default="val", choices=list(SPLITS))
    ap.add_argument("--asm", nargs="*", default=[], help="override assembly settings, key=value")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--compare", default=None, help="another registry name: paired scene bootstrap on the same frames")
    ap.add_argument("--tag", default="")
    ap.add_argument("--test-id", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--expect", nargs="*", default=[], help="metric=value pairs that must be reproduced within 0.0005 (exit 1 otherwise): layer 1 of the protocol")
    a = ap.parse_args()
    reg = registry()
    if a.list or not a.name:
        for k, v in reg.items():
            print(f"{k:18} {v['kind']:8} {v.get('run', '')} {v.get('set', '')} {v.get('what', '')}")
        return
    if a.split == "test" and not test_allowed(a.test_id):
        sys.exit("TEST refused: pass --test-id <run id> that is logged in evaluation/TEST_USAGE_LOG.md (select and tune on VAL only)")
    import sparse_metrics as SM
    over = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    g, gts, info = graphs_of(a.name, reg[a.name], a.split, over, a.tau)
    t0 = time.time()
    m = SM.card_any(g, gts, extra=True)
    info["metric_s"] = time.time() - t0
    card = dict(name=a.name, split=a.split, frames=len(gts), registry=reg[a.name], overrides=dict(asm=over, tau=a.tau), info=info, commit=commit(),
                date=time.strftime("%Y-%m-%d %H:%M"), metrics=m)
    if a.compare:
        import bootstrap as SB
        import topo_metrics as TM
        g2, gts2, _ = graphs_of(a.compare, reg[a.compare], a.split)
        assert len(g2) == len(g) and all(len(x["pts"]) == len(y["pts"]) for x, y in zip(gts, gts2)), "frame sets differ"
        sc = scenes(a.split)
        pr = SB.paired(SB.frame_records(g, gts), SB.frame_records(g2, gts), sc, B=1000)
        pr.update(TM.paired([TM.frame_record(p, q) for p, q in zip(g, gts)], [TM.frame_record(p, q) for p, q in zip(g2, gts)], sc, B=1000))
        card["compare"] = dict(name=a.compare, metrics=SM.card_any(g2, gts, extra=True), paired=pr)
    os.makedirs(f"{R}/runs/eval/{a.name}", exist_ok=True)
    out = f"{R}/runs/eval/{a.name}/{a.split}{'_' + a.tag if a.tag else ''}.json"
    json.dump(card, open(out, "w", encoding="utf-8"), indent=1, default=float)
    print(f"{a.name} on {a.split} ({len(gts)} frames), commit {card['commit']}, assembly {info['asm_ms']:.1f} ms per frame (CPU), metrics {info['metric_s']:.0f} s")
    print("  " + "  ".join(f"{k} {m[k]:.3f}" for k in SHOW if k in m))
    if a.compare:
        c2 = card["compare"]["metrics"]
        print(f"{a.compare}:")
        print("  " + "  ".join(f"{k} {c2[k]:.3f}" for k in SHOW if k in c2))
        p = card["compare"]["paired"]
        keys = [k for k in ("S", "Q", "F1@1.0", "J", "lanes_found", "topo_F1", "R@1_far", "dup_frac") if k in p]
        print(f"{a.name} - {a.compare} (95 % over scenes): " + "  ".join(f"{k} {p[k]['diff']:+.3f} [{p[k]['lo']:+.3f}, {p[k]['hi']:+.3f}]" for k in keys if isinstance(p.get(k), dict) and "diff" in p[k]))
        if "verdict" in p.get("S", {}):
            print("verdict on S (band 0.010):", p["S"]["verdict"])
    print("card:", out)
    bad = [f"{k} {m.get(k, float('nan')):.4f} != {v}" for k, v in (x.split("=") for x in a.expect) if not abs(m.get(k, float("nan")) - float(v)) <= 5e-4]
    if bad:
        sys.exit("stored numbers NOT reproduced: " + "; ".join(bad))
    if a.expect:
        print("stored numbers reproduced:", " ".join(a.expect))


if __name__ == "__main__":
    main()
