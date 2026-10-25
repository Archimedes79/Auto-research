"""The one evaluation entry point: a named model (models/registry.json) on a split, with its standard settings, all metrics, one card.

    tools/env/_py14.sh evaluate.py champion                       # VAL (873 frames), writes runs/eval/champion/val.json
    tools/env/_py14.sh evaluate.py fast --compare champion        # + paired scene bootstrap against another name (same frames)
    tools/env/_py14.sh evaluate.py champion --asm fork_prune=0.4  # try a setting (recorded in the card as an override)
    tools/env/_py14.sh evaluate.py champion --split test --test-id E230.f1p   # TEST only when the id is logged in evaluation/TEST_USAGE_LOG.md
    tools/env/_py14.sh evaluate.py champion --subset val250 --fresh           # the network path: 250 VAL frames re-inferred on the GPU, no cache
    tools/env/_py14.sh evaluate.py --golden check [--cached-only]            # layer 1 of the protocol: every golden number reproduced (exit 1 otherwise)
    tools/env/_py14.sh evaluate.py --golden write [names ...]                # (re)write golden entries: all registry names on VAL from the cache
    tools/env/_py14.sh evaluate.py --golden write champion --subset val250 --fresh --tol 0.002   # the uncached entry (idle GPU)
    tools/env/_py14.sh evaluate.py --list

What it does
  * builder: the raw network outputs of (run, split, M, K0) are computed once on the GPU and cached in ~/exp_gcar_b123/eval_cache/ (the
    loop-18 caches ~/exp_gcar_b123/p18/raw_val873_<run>_<M>.pkl are reused); assembly settings and thresholds are re-scored on the CPU in seconds.
    --fresh ignores and does not write the cache (the network path is exercised); --subset val250 takes the fixed 250-frame VAL subset of the
    triage rungs (p16_common.val_subset)
  * graphs: stored graphs of a system that is not re-run (the control E94)
  * metrics: sparse_metrics.card_any(extra=True) - S, Q, F1, J, FL, vertices per 100 m, lanes found, topo P / R / F1, dup_frac, recall by range
  * card: runs/eval/<name>/<split>[_<subset>][_fresh][_<tag>].json with the registry entry, overrides, frames, cache file, code commit, metrics, assembly ms per frame
  * golden: evaluation/golden.json holds one entry per (name, split, subset, fresh) with the metrics of GOLD_KEYS rounded to 4 decimals, the
    commit and the tolerance; `--golden check` re-evaluates every entry and compares (the check writes cards with the tag `golden`, never the
    standard card); a golden entry changes only together with a ledger row that says why (a metric fix, a new champion)
Timing of the network (ms per frame on an idle GPU) is not measured here: evaluation/drivers/p17_latency.py (protocol in PROTOCOL.md).
"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

R = BP.ROOT
HOME = os.path.expanduser("~")
CACHE = f"{HOME}/exp_gcar_b123/eval_cache"
GOLDEN = f"{R}/evaluation/golden.json"
SPLITS = {"val": ("val_clean", "val"), "test": ("test_clean", "test")}
SHOW = ("S", "Q", "F1@1.0", "J", "topo_F1", "lanes_found", "R@1_far", "false_len_frac", "dup_frac", "vert100")
GOLD_KEYS = ("S", "Q", "F1@1.0", "J", "topo_F1", "lanes_found", "R@1_far", "vert100")
TOL = 5e-4  # cached re-scoring is deterministic: 4 decimals must agree


def registry():
    return {k: v for k, v in json.load(open(f"{R}/models/registry.json", encoding="utf-8")).items() if not k.startswith("_")}


def commit():
    try:
        h = subprocess.run(["git", "-c", "safe.directory=*", "-C", R, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-c", "safe.directory=*", "-C", R, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10).stdout.strip()
        return (h + ("+dirty" if dirty else "")) if h else "no-git"
    except Exception:
        return "no-git"


def test_allowed(test_id):
    log = open(f"{R}/evaluation/TEST_USAGE_LOG.md", encoding="utf-8").read()
    return bool(test_id) and test_id in log


def subset_ks(split, subset):
    """frame positions of a named subset of the split (None = all frames)"""
    if not subset:
        return None
    if split == "val" and subset == "val250":
        import p16_common as PC
        return PC.val_subset(250)
    sys.exit(f"unknown subset {subset} for split {split} (known: val250 on val)")


def builder_raw(e, split, ks=None, fresh=False):
    """(raw node outputs, ground truths, source) of the frames ks of the split (None = all); cached unless ks is given or fresh is set"""
    run, M, K0 = e["run"], e["set"]["M"], e["set"]["K0"]
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/raw_{split}_{run}_{M}_{K0}.pkl"
    cached = ks is None and not fresh
    if cached:
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
    ds = PD.BuilderData(SPLITS[split][0], ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, list(range(len(ds))), taus=(e["tau"],), bs=8, workers=2, raw=raw)
    if cached:
        pickle.dump((raw, gts), open(path, "wb"))
        return raw, gts, path
    return raw, gts, "fresh (no cache)"


def graphs_of(name, e, split, asm_over=None, tau=None, ks=None, fresh=False):
    """graphs, ground truths, info for a registry entry"""
    if e["kind"] == "graphs":
        z = pickle.load(open(f"{R}/{e['files'][split]}", "rb"))
        import p16_data as PD
        ds = PD.BuilderData(SPLITS[split][0], ks=None, train=False)
        want = None if ks is None else set(int(x) for x in ks)
        keep = [i for i, k in enumerate(z["ks"]) if want is None or int(k) in want]
        gts = [ds.gt[ds.rows[int(z["ks"][i])]] for i in keep]
        return [z["preds"][i] for i in keep], gts, dict(source=e["files"][split], asm_ms=float("nan"))
    import p16_infer as PI
    import p16_train as PT
    raw, gts, cache = builder_raw(e, split, ks, fresh)
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


def scenes(split, ks=None):
    sp = json.load(open(f"{R}/data/splits.json"))
    rows = np.array(sp[SPLITS[split][1]])
    if ks is not None:
        rows = rows[np.asarray(ks)]
    return np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["scene"][rows]


def evaluate(name, reg, split="val", over=None, tau=None, subset=None, fresh=False, compare=None, tag=""):
    """one card: the registry name on the split (or its subset), written to runs/eval/<name>/; returns the card"""
    import sparse_metrics as SM
    over = over or {}
    ks = subset_ks(split, subset)
    g, gts, info = graphs_of(name, reg[name], split, over, tau, ks, fresh)
    t0 = time.time()
    m = SM.card_any(g, gts, extra=True)
    info["metric_s"] = time.time() - t0
    card = dict(name=name, split=split, subset=subset or "", fresh=bool(fresh), frames=len(gts), registry=reg[name], overrides=dict(asm=over, tau=tau),
                info=info, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"), metrics=m)
    if compare:
        import bootstrap as SB
        import topo_metrics as TM
        g2, gts2, _ = graphs_of(compare, reg[compare], split, None, None, ks, False)
        assert len(g2) == len(g) and all(len(x["pts"]) == len(y["pts"]) for x, y in zip(gts, gts2)), "frame sets differ"
        sc = scenes(split, ks)
        pr = SB.paired(SB.frame_records(g, gts), SB.frame_records(g2, gts), sc, B=1000)
        pr.update(TM.paired([TM.frame_record(p, q) for p, q in zip(g, gts)], [TM.frame_record(p, q) for p, q in zip(g2, gts)], sc, B=1000))
        card["compare"] = dict(name=compare, metrics=SM.card_any(g2, gts, extra=True), paired=pr)
    os.makedirs(f"{R}/runs/eval/{name}", exist_ok=True)
    out = f"{R}/runs/eval/{name}/{split}{'_' + subset if subset else ''}{'_fresh' if fresh else ''}{'_' + tag if tag else ''}.json"
    json.dump(card, open(out, "w", encoding="utf-8"), indent=1, default=float)
    card["path"] = out
    return card


def show(card):
    m = card["metrics"]
    what = f"{card['name']} on {card['split']}{' ' + card['subset'] if card['subset'] else ''}{' fresh' if card['fresh'] else ''}"
    print(f"{what} ({card['frames']} frames), commit {card['commit']}, assembly {card['info']['asm_ms']:.1f} ms per frame (CPU), metrics {card['info']['metric_s']:.0f} s")
    print("  " + "  ".join(f"{k} {m[k]:.3f}" for k in SHOW if k in m))
    if "compare" in card:
        c2, cmp = card["compare"]["metrics"], card["compare"]["name"]
        print(f"{cmp}:")
        print("  " + "  ".join(f"{k} {c2[k]:.3f}" for k in SHOW if k in c2))
        p = card["compare"]["paired"]
        keys = [k for k in ("S", "Q", "F1@1.0", "J", "lanes_found", "topo_F1", "R@1_far", "dup_frac") if k in p]
        print(f"{card['name']} - {cmp} (95 % over scenes): " + "  ".join(f"{k} {p[k]['diff']:+.3f} [{p[k]['lo']:+.3f}, {p[k]['hi']:+.3f}]" for k in keys if isinstance(p.get(k), dict) and "diff" in p[k]))
        if "verdict" in p.get("S", {}):
            print("verdict on S (band 0.010):", p["S"]["verdict"])
    print("card:", card["path"])


# ------------------------------------------------------------------ golden numbers (layer 1 of the protocol)


def golden_load():
    if os.path.exists(GOLDEN):
        return json.load(open(GOLDEN, encoding="utf-8"))
    return dict(_about="Golden numbers of evaluate.py: one entry per (name, split, subset, fresh). `evaluate.py --golden check` must reproduce every entry "
                       "within its tolerance; an entry changes only together with a ledger row that says why.", entries=[])


def golden_key(e):
    return (e["name"], e["split"], e.get("subset", ""), bool(e.get("fresh")))


def golden_write(names, reg, split, subset, fresh, tol):
    g = golden_load()
    for name in names:
        card = evaluate(name, reg, split, subset=subset, fresh=fresh, tag="golden")
        show(card)
        ent = dict(name=name, split=split, subset=subset or "", fresh=bool(fresh), frames=card["frames"], tol=tol, commit=card["commit"], date=card["date"],
                   metrics={k: round(float(card["metrics"][k]), 4) for k in GOLD_KEYS if k in card["metrics"]})
        g["entries"] = [x for x in g["entries"] if golden_key(x) != golden_key(ent)] + [ent]
    g["entries"].sort(key=lambda x: (x["split"], x["subset"], x["fresh"], x["name"]))
    json.dump(g, open(GOLDEN, "w", encoding="utf-8"), indent=1)
    print(f"golden: {len(g['entries'])} entries in {os.path.relpath(GOLDEN, R)}")


def golden_check(reg, cached_only=False, fresh_only=False):
    g = golden_load()
    bad, n = [], 0
    for ent in g["entries"]:
        label = f"{ent['name']} {ent['split']}{' ' + ent['subset'] if ent['subset'] else ''}{' fresh' if ent['fresh'] else ''}"
        if (cached_only and ent["fresh"]) or (fresh_only and not ent["fresh"]):
            print(f"skip  {label}")
            continue
        if ent["name"] not in reg:
            bad.append(f"{label}: not in the registry")
            continue
        card = evaluate(ent["name"], reg, ent["split"], subset=ent["subset"] or None, fresh=ent["fresh"], tag="golden")
        n += 1
        miss = [f"{k} {card['metrics'].get(k, float('nan')):.4f} != {v}" for k, v in ent["metrics"].items() if not abs(card["metrics"].get(k, float("nan")) - v) <= ent["tol"]]
        if card["frames"] != ent["frames"]:
            miss.append(f"frames {card['frames']} != {ent['frames']}")
        if miss:
            bad.append(f"{label}: " + "; ".join(miss))
            print(f"MISS  {label}: " + "; ".join(miss))
        else:
            print(f"ok    {label} ({card['frames']} frames, tol {ent['tol']}): " + " ".join(f"{k} {v}" for k, v in ent["metrics"].items()))
    if bad:
        sys.exit(f"golden numbers NOT reproduced ({len(bad)} of {n}): " + " | ".join(bad))
    print(f"golden numbers reproduced: {n} entries")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="*")
    ap.add_argument("--split", default="val", choices=list(SPLITS))
    ap.add_argument("--subset", default="", help="named frame subset of the split (val250)")
    ap.add_argument("--fresh", action="store_true", help="ignore and do not write the network-output cache (the network path runs on the GPU)")
    ap.add_argument("--asm", nargs="*", default=[], help="override assembly settings, key=value")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--compare", default=None, help="another registry name: paired scene bootstrap on the same frames")
    ap.add_argument("--tag", default="")
    ap.add_argument("--test-id", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--expect", nargs="*", default=[], help=f"metric=value pairs that must be reproduced within {TOL} (exit 1 otherwise)")
    ap.add_argument("--golden", choices=["check", "write"], help="check: reproduce every entry of evaluation/golden.json; write: (re)write the entries of the names given (default: all registry names)")
    ap.add_argument("--cached-only", action="store_true", help="--golden check without the fresh (GPU) entries")
    ap.add_argument("--fresh-only", action="store_true", help="--golden check of the fresh (GPU) entries only")
    ap.add_argument("--tol", type=float, default=TOL, help="tolerance stored with a golden entry on --golden write")
    a = ap.parse_args()
    reg = registry()
    if a.list or (not a.name and not a.golden):
        for k, v in reg.items():
            print(f"{k:18} {v['kind']:8} {v.get('run', '')} {v.get('set', '')} {v.get('what', '')}")
        return
    if a.split == "test" and not test_allowed(a.test_id):
        sys.exit("TEST refused: pass --test-id <run id> that is logged in evaluation/TEST_USAGE_LOG.md (select and tune on VAL only)")
    if a.golden == "check":
        return golden_check(reg, a.cached_only, a.fresh_only)
    if a.golden == "write":
        return golden_write(a.name or list(reg), reg, a.split, a.subset or None, a.fresh, a.tol)
    if len(a.name) != 1:
        sys.exit("one registry name (or --golden / --list)")
    over = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    card = evaluate(a.name[0], reg, a.split, over, a.tau, a.subset or None, a.fresh, a.compare, a.tag)
    show(card)
    m = card["metrics"]
    bad = [f"{k} {m.get(k, float('nan')):.4f} != {v}" for k, v in (x.split("=") for x in a.expect) if not abs(m.get(k, float("nan")) - float(v)) <= TOL]
    if bad:
        sys.exit("stored numbers NOT reproduced: " + "; ".join(bad))
    if a.expect:
        print("stored numbers reproduced:", " ".join(a.expect))


if __name__ == "__main__":
    main()
