"""The one evaluation entry point: a named model (models/registry.json) on a split, with its standard settings, all metrics, one card.

    tools/env/_py14.sh evaluate.py champion                       # VAL (873 frames), writes runs/eval/champion/val.json
    tools/env/_py14.sh evaluate.py fast --compare champion        # + paired scene bootstrap against another name (same frames)
    tools/env/_py14.sh evaluate.py champion --asm fork_prune=0.4  # try a setting (recorded in the card as an override)
    tools/env/_py14.sh evaluate.py champion --split test --test-id E230.f1p   # TEST only when the id is logged in evaluation/TEST_USAGE_LOG.md
    tools/env/_py14.sh evaluate.py champion --subset val250 --fresh           # the network path: 250 VAL frames re-inferred on the GPU, no cache
    tools/env/_py14.sh evaluate.py --golden check [--cached-only]            # layer 1 of the protocol: every golden number reproduced (exit 1 otherwise); golden.json holds 7 entries, not every registry name (PROTOCOL.md)
    tools/env/_py14.sh evaluate.py --golden write [names ...]                # (re)write golden entries of the names given (default: all registry names; protected file: a grant under a decided D<n> first)
    tools/env/_py14.sh evaluate.py --golden write champion --subset val250 --fresh --tol 0.0005   # the uncached entry (idle GPU; the stored tolerance)
    tools/env/_py14.sh evaluate.py --list
    tools/env/_py14.sh evaluate.py champion --route --failure dark,blur,cam_front3_off --timing   # parts (eval_parts.py): route success, failure series, ms per frame (GPU); card tag `parts`
    tools/env/_py14.sh evaluate.py champion --route --route-tokens tok_p18_f1pr_val_clean         # route success from stored tokens (CPU)
    tools/env/_py14.sh evaluate.py champion --failure dark,blur --series-dir e130_champion         # failure graph metrics per phase from stored series graphs (CPU)

What it does
  * builder: the raw network outputs of (run, split, M, K0, every other `set` field) are computed once on the GPU and cached in
    ~/exp_gcar_b123/eval_cache/ (cache_paths; the loop-18 caches ~/exp_gcar_b123/p18/raw_val873_<run>_<M>.pkl are reused for entries without
    other settings); assembly settings and thresholds are re-scored on the CPU in seconds.
    --fresh ignores and does not write the cache (the network path is exercised); --subset val250 takes the fixed 250-frame VAL subset of the
    triage rungs (p16_common.val_subset), from the cache of the whole split when there is one (the network runs only without a cache or with --fresh)
  * graphs: stored graphs of a system that is not re-run (the control E94)
  * union: `members` = registry names of builders; their raw trees (each read from its own cache) are joined per frame (p16_infer.union_raw) and
    assembled once with the union entry's tau / asm (E264, output ensemble)
  * metrics: sparse_metrics.card_any(extra=True) - S, Q, F1, J, FL, vertices per 100 m, lanes found, topo P / R / F1, dup_frac, recall by range
  * card: runs/eval/<name>/<split>[_<subset>][_fresh][_<tag>].json (the contract ../autoresearch/schemas/card.schema.json, schema 1) with the registry entry, overrides, frames, cache file, code commit, metrics, assembly ms per frame
  * golden: evaluation/golden.json holds one entry per (name, split, subset, fresh) with the metrics of GOLD_KEYS rounded to 4 decimals, the
    commit and the tolerance; `--golden check` re-evaluates every entry and compares (the check writes cards with the tag `golden_check`, which git
    ignores; `--golden write` writes the committed `golden` card, never the standard card); a golden entry changes only together with a ledger row that says why (a metric fix, a new champion)
  * parts (W11, eval_parts.py): --route (route success), --failure (graph metrics per phase of a failure series, with --route also the route success per phase),
    --timing (ms per frame of the whole chain, drivers/p18_latency.py on an idle GPU; protocol in PROTOCOL.md); they call the code of the loop drivers, run the network
    unless a stored input is named (--route-tokens, --series-dir), and join the card as `metrics` (keys in eval_parts.py), `better` and `info.parts`; without --tag the
    card is written as <split>_parts.json, never over the standard card
"""
import bflc_paths as BP  # project paths from the location of this checkout
import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np

import eval_parts as EP  # route success, failure series, timing (parts of a card)

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
    """short commit of the checkout, '+dirty' when tracked files differ ('+?' when the status check times out: git status on /mnt/c can take long)"""
    try:
        h = subprocess.run(["git", "-c", "safe.directory=*", "-C", R, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "no-git"
    if not h:
        return "no-git"
    try:
        dirty = subprocess.run(["git", "-c", "safe.directory=*", "-C", R, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=120).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return h + "+?"


def test_allowed(test_id, log=None):
    """True if test_id is a run id (E<number>...) and the TEST usage log names it as a whole id, the rule of autoresearch/tools/run.py
    (review bug L2, D24: a substring test let `E` pass; the run-id shape keeps date parts such as `2026` or `10` and words out)"""
    if not test_id or not re.fullmatch(r"E\d+[A-Za-z0-9_.-]*", test_id):
        return False
    if log is None:
        log = open(f"{R}/evaluation/TEST_USAGE_LOG.md", encoding="utf-8").read()
    return re.search(r"(?<![\w.])" + re.escape(test_id) + r"(?![\w.])", log) is not None


def s_comparable(a, b):
    """False when exactly one of the two names is a per-piece-DP entry (`*_dp`): sparse_metrics.Sp divides the oracle 14.3 by the node-id count (vert100), which
    simplify_dp lowers, so a `*_dp` entry gains S without a better graph (D25: recall 0.7675 -> recall_dp 0.7789 in S = +0.0115 above the band, Q 0.7404 -> 0.7399);
    across that basis Q is the number to read"""
    return a.endswith("_dp") == b.endswith("_dp")


def subset_ks(split, subset):
    """frame positions of a named subset of the split (None = all frames)"""
    if not subset:
        return None
    if split == "val" and subset == "val250":
        import p16_common as PC
        return PC.val_subset(250)
    sys.exit(f"unknown subset {subset} for split {split} (known: val250 on val)")


def cache_paths(e, split):
    """raw-output cache files of a builder entry on the split: [the file written and read first, older names also read]. The name holds run, M, K0
    and, when the entry has any other setting that changes the network output (another `set` field, a design other than D7), a digest of those
    settings; the older names (the loop-18 files without K0, all made with K0 = M / 2) belong to entries without such settings and with K0 = M / 2
    only, so a new setting never re-scores an old cache"""
    run, M, K0 = e["run"], e["set"]["M"], e["set"]["K0"]
    other = {k: v for k, v in e["set"].items() if k not in ("M", "K0")}
    if e["design"] != "D7":
        other["design"] = e["design"]
    if other:
        return [f"{CACHE}/raw_{split}_{run}_{M}_{K0}_{digest(other)}.pkl"]
    old = [f"{HOME}/exp_gcar_b123/p18/raw_val873_{run}_{M}.pkl"] + ([f"{HOME}/exp_gcar_b123/p18/raw_val873.pkl"] if run == "e123_D7_r3_f1" and M == 288 else [])
    return [f"{CACHE}/raw_{split}_{run}_{M}_{K0}.pkl"] + (old if split == "val" and K0 == M // 2 else [])


def entry_cfg(e):
    """the builder config overrides of a builder entry: its run card's cfg, then the entry's `set`"""
    import p16_train as PT
    extra = dict(json.load(open(f"{R}/runs/e123_ladder/cards/{e['run']}.json", encoding="utf-8-sig"))["cfg"])
    extra.update(PT.parse_set([f"{k}={v}" for k, v in e["set"].items()]))
    return extra


def build_net(e, dev="cuda"):
    """(network, config) of a builder entry with its trained weights, on dev"""
    import torch
    import p16_train as PT
    net, cfg = PT.build(e["design"], entry_cfg(e))
    net = net.to(dev)
    net.load_state_dict(torch.load(f"{R}/runs/e123/{e['run']}/model.pt", map_location=dev), strict=False)
    return net, cfg


def builder_raw(e, split, ks=None, fresh=False, require_cache=False, dev="cuda"):
    """(raw node outputs, ground truths, source) of the frames ks of the split (None = all). Unless fresh is set they come from the cache of the
    whole split (cache_paths; frames ks taken from it); without a cache the network runs on dev, and only a whole split computed on the GPU
    (the device of every cache so far) is written"""
    paths = cache_paths(e, split)
    if not fresh:
        for p in paths:
            if os.path.exists(p):
                raw, gts = pickle.load(open(p, "rb"))
                if ks is None:
                    return raw, gts, p
                return [raw[int(k)] for k in ks], [gts[int(k)] for k in ks], f"{p} [{len(ks)} frames]"
    if require_cache:
        raise RuntimeError(f"no cached network outputs for {e['run']} {e['set']} on {split} (would need the GPU)")
    import p16_data as PD
    import p16_infer as PI
    net, cfg = build_net(e, dev)
    ds = PD.BuilderData(SPLITS[split][0], ks=ks, train=False, with_prev=cfg.temporal)
    raw = []
    _, gts, _ = PI.predict(net, ds, list(range(len(ds))), taus=(e["tau"],), bs=8, workers=2, raw=raw, dev=dev)
    if ks is None and not fresh and dev == "cuda":
        os.makedirs(CACHE, exist_ok=True)
        tmp = f"{paths[0]}.tmp{os.getpid()}"  # written whole, then renamed: a crash never leaves a truncated cache under the real name
        try:
            with open(tmp, "wb") as f:
                pickle.dump((raw, gts), f)
            os.replace(tmp, paths[0])
        except BaseException:  # a failed or interrupted write leaves no temp file behind
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        return raw, gts, paths[0]
    return raw, gts, "fresh (no cache)"


def graphs_of(name, e, split, asm_over=None, tau=None, ks=None, fresh=False, require_cache=False):
    """graphs, ground truths, info for a registry entry (kind graphs: stored; builder / union: raw_of, assembled with the entry's tau / asm)"""
    if e["kind"] == "graphs":
        z = pickle.load(open(f"{R}/{e['files'][split]}", "rb"))
        import p16_data as PD
        ds = PD.BuilderData(SPLITS[split][0], ks=None, train=False)
        want = None if ks is None else set(int(x) for x in ks)
        keep = [i for i, k in enumerate(z["ks"]) if want is None or int(k) in want]
        gts = [ds.gt[ds.rows[int(z["ks"][i])]] for i in keep]
        return [z["preds"][i] for i in keep], gts, dict(source=e["files"][split], asm_ms=float("nan"))
    import p16_infer as PI
    raw, gts, info, base = raw_of(e, split, ks, fresh, require_cache)
    asm = dict(e["asm"])
    asm.update(asm_over or {})
    t, cfg = (e["tau"] if tau is None else tau), builder_cfg(base)
    t0 = time.time()
    g = [PI.assemble(r, None, t, cfg, **asm) for r in raw]
    return g, gts, dict(info, asm=asm, tau=t, asm_ms=(time.time() - t0) / max(len(gts), 1) * 1e3)


def raw_of(e, split, ks=None, fresh=False, require_cache=False):
    """(raw trees, ground truths, info, the builder entry whose config assembles them) of a builder or union entry. Union (E264): the trees of
    the member entries (each from its own cache) joined per frame, one joined frame at a time (memory); the members must share the ground truth"""
    if e["kind"] != "union":
        raw, gts, cache = builder_raw(e, split, ks, fresh, require_cache)
        return raw, gts, dict(cache=cache), e
    import p16_infer as PI
    reg = registry()
    members = [reg[m] for m in e["members"]]
    got = [builder_raw(m, split, ks, fresh, require_cache) for m in members]
    if len({digest(x[1]) for x in got}) != 1:
        raise ValueError(f"the members {e['members']} were computed on different frames or ground truths")
    raw = (PI.union_raw(rs) for rs in zip(*(x[0] for x in got)))
    return raw, got[0][1], dict(cache=[x[2] for x in got], members={m: reg[m] for m in e["members"]}), members[0]


def builder_cfg(e):
    """the builder config of a builder entry without building the network (the config of p16_train.build)"""
    import p16_builder as PB
    import p16_train as PT
    kw = dict(PT.DESIGNS[e["design"]])
    kw.update(entry_cfg(e))
    return PB.DefaultCfg(**{k: v for k, v in kw.items() if k not in ("w",)})


def scenes(split, ks=None):
    sp = json.load(open(f"{R}/data/splits.json"))
    rows = np.array(sp[SPLITS[split][1]])
    if ks is not None:
        rows = rows[np.asarray(ks)]
    return np.load(f"{HOME}/exp_gcar_b123/poses_val.npz")["scene"][rows]


def digest(obj, h=None):
    """12-hex sha1 of nested dicts / lists / numpy arrays / scalars: the same content gives the same digest whatever the container types (the card's
    protocol: the ground truth the frames were scored against, so decide.py refuses a comparison across ground truths; also in the names of the
    network-output caches, cache_paths). FROZEN: a change of the formula splits every comparison across the cards written before it and renames
    those caches, so a different formula is a new function"""
    top = h is None
    h = h or hashlib.sha1()
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            h.update(str(k).encode() + b"\0")
            digest(obj[k], h)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for x in obj:
            digest(x, h)
        h.update(b"]")
    elif isinstance(obj, np.ndarray) and obj.dtype != object:
        h.update(f"{obj.dtype.str}{obj.shape}".encode() + np.ascontiguousarray(obj).tobytes() + b"\0")
    elif isinstance(obj, np.generic):
        h.update(repr(obj.item()).encode() + b"\0")
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        h.update(repr(obj).encode() + b"\0")
    else:  # an object whose repr may carry a memory address would make the digest differ from run to run: refuse instead
        raise TypeError(f"cannot digest a {type(obj).__name__}")
    return h.hexdigest()[:12] if top else None


def seed_of(entry):
    """the training seed of a registry entry: its own `seed` (written where the run card is a copy with another seed: e24_j1phd_s0_at50), else the seed of
    its run card (runs/e123_ladder/cards/<run>.json); None when none is recorded (stored graphs)"""
    own = entry.get("seed")
    if isinstance(own, (int, str)) and not isinstance(own, bool):
        return own
    try:
        s = json.load(open(f"{R}/runs/e123_ladder/cards/{entry['run']}.json", encoding="utf-8-sig")).get("seed")
    except (OSError, ValueError, KeyError):
        return None
    return s if isinstance(s, (int, str)) and not isinstance(s, bool) else None


def effective(e, over, tau):
    """the registry entry with the overrides of the card applied (--asm, --tau)"""
    return dict(e, asm={**e.get("asm", {}), **(over or {})}, tau=e.get("tau") if tau is None else tau)


def evaluate(name, reg, split="val", over=None, tau=None, subset=None, fresh=False, compare=None, tag="", require_cache=False, parts=None):
    """one card: the registry name on the split (or its subset), written to runs/eval/<name>/; returns the card. parts = {route, failure, timing} options of
    eval_parts.run: their numbers join the card (metrics, better, info.parts); they run the network (GPU) except from stored inputs, on the whole split"""
    import sparse_metrics as SM
    over = over or {}
    ks = subset_ks(split, subset)
    if parts and (subset or fresh):
        raise ValueError("route / failure / timing run on the whole split, without --subset and --fresh")
    g, gts, info = graphs_of(name, reg[name], split, over, tau, ks, fresh, require_cache)
    protocol = dict(gt_sha=digest(gts))  # the ground truth scored against (before the minute of scoring: a ground truth that cannot be digested fails early)
    t0 = time.time()
    m = SM.card_any(g, gts, extra=True)
    info["metric_s"] = time.time() - t0
    better = SM.better(m)  # each metric's direction (card contract)
    if parts:  # last: the timing wants a machine that does nothing else
        add = EP.run(effective(reg[name], over, tau), reg, split, parts, build_net)
        m.update(add["metrics"])
        better.update(add["better"])
        info["parts"] = add["info"]
    nan = [k for k, v in m.items() if isinstance(v, float) and v != v]
    if nan:  # the contract wants numbers: a metric without a value is named in info, not written as NaN
        info["nan_metrics"] = nan
        m = {k: v for k, v in m.items() if k not in nan}
        better = {k: v for k, v in better.items() if k in m}
    card = dict(schema=1, name=name, split=split, subset=subset or "", fresh=bool(fresh), frames=len(gts), registry=reg[name], overrides=dict(asm=over, tau=tau),
                info=info, commit=commit(), date=time.strftime("%Y-%m-%d %H:%M"), metrics=m, better=better,
                protocol=protocol)  # decide.py refuses cards of one comparison that differ in it
    seed = seed_of(reg[name])
    if seed is not None:
        card["seed"] = seed  # decide.py pairs the cards of two arms by it
    if compare:
        import bootstrap as SB
        import topo_metrics as TM
        g2, gts2, _ = graphs_of(compare, reg[compare], split, None, None, ks, False)
        assert len(g2) == len(g) and all(len(x["pts"]) == len(y["pts"]) for x, y in zip(gts, gts2)), "frame sets differ"
        assert digest(gts2) == protocol["gt_sha"], "the control was scored against another ground truth"
        sc = scenes(split, ks)
        pr = SB.paired(SB.frame_records(g, gts), SB.frame_records(g2, gts), sc, B=1000)
        pr.update(TM.paired([TM.frame_record(p, q) for p, q in zip(g, gts)], [TM.frame_record(p, q) for p, q in zip(g2, gts)], sc, B=1000))
        m2 = SM.card_any(g2, gts, extra=True)
        card["compare"] = dict(name=compare, metrics=m2, better=SM.better(m2), paired=pr, s_not_comparable=not s_comparable(name, compare))
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
    parts = [k for k in m if k.startswith(("route_", "ms_per_frame")) or (k.startswith("series.") and k.endswith((".S", ".Q", ".route_success")))]
    if parts:  # the parts of eval_parts: route success, the failure series per phase, the time of the chain
        print("  " + "  ".join(f"{k} {m[k]:.3f}" for k in parts))
    if "compare" in card:
        c2, cmp = card["compare"]["metrics"], card["compare"]["name"]
        print(f"{cmp}:")
        print("  " + "  ".join(f"{k} {c2[k]:.3f}" for k in SHOW if k in c2))
        p = card["compare"]["paired"]
        keys = [k for k in ("S", "Q", "F1@1.0", "J", "lanes_found", "topo_F1", "R@1_far", "dup_frac") if k in p]
        print(f"{card['name']} - {cmp} (95 % over scenes): " + "  ".join(f"{k} {p[k]['diff']:+.3f} [{p[k]['lo']:+.3f}, {p[k]['hi']:+.3f}]" for k in keys if isinstance(p.get(k), dict) and "diff" in p[k]))
        if card["compare"].get("s_not_comparable"):
            print(f"WARNING: S is NOT comparable between {card['name']} and {cmp}: exactly one of them is a `_dp` entry, and S counts the node-id vertices that simplify_dp lowers (D25). "
                  "Read Q; the S verdict below says nothing.")
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


def golden_check(reg, cached_only=False, fresh_only=False, test_id=""):
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
        if ent["split"] == "test" and not test_allowed(test_id):
            print(f"skip  {label} (TEST: pass a --test-id logged in TEST_USAGE_LOG.md)")
            continue
        try:
            card = evaluate(ent["name"], reg, ent["split"], subset=ent["subset"] or None, fresh=ent["fresh"], tag="golden_check", require_cache=cached_only)  # the check never rewrites the committed golden card
        except RuntimeError as e:
            bad.append(f"{label}: {e}")
            print(f"MISS  {label}: {e}")
            continue
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
    ap.add_argument("--route", action="store_true", help="part: route success on the split (builder -> assembly -> route input -> route head; GPU, or CPU with --route-tokens)")
    ap.add_argument("--route-tokens", default="", help="stored route tokens (~/exp_gcar_b123/p10/<name>.pkl): route success from them, no builder run (CPU)")
    ap.add_argument("--route-head", default=EP.ROUTE_HEAD, help="route head (runs/<name>/model.pt); a route comparison needs seeds s0 and s1")
    ap.add_argument("--route-step", type=float, default=4.0, help="route input: points every <= step m of an edge")
    ap.add_argument("--failure", default="", help="part: failure series, perturbations separated by commas (dark,blur,cam_front3_off,...); needs the series input of p17_series.py (GPU) or --series-dir")
    ap.add_argument("--variant", default="single", help="input variant of the series (single, maxdecay_0.95, ema_hold_mon_0.5, gate, ...)")
    ap.add_argument("--series-dir", default="", help="read the stored graphs of a series run (runs/e17/<dir>/<pert>_<variant><suffix>.pkl) instead of running the builder (CPU)")
    ap.add_argument("--series-suffix", default="", help="file name suffix of the stored series graphs (e.g. _fp03)")
    ap.add_argument("--series-bs", type=int, default=4)
    ap.add_argument("--series-workers", type=int, default=2, help="data loader workers of the series run")
    ap.add_argument("--timing", action="store_true", help="part: ms per frame of the whole chain (drivers/p18_latency.py), IDLE GPU only, nothing else running")
    ap.add_argument("--timing-n", type=int, default=60, help="frames timed")
    a = ap.parse_args()
    reg = registry()
    if a.list or (not a.name and not a.golden):
        for k, v in reg.items():
            print(f"{k:18} {v['kind']:8} {v.get('run', ' + '.join(v.get('members', [])))} {v.get('set', '')} {v.get('what', '')}")
        return
    if a.split == "test" and a.golden != "check" and not test_allowed(a.test_id):
        sys.exit("TEST refused: pass --test-id <run id> that is logged in evaluation/TEST_USAGE_LOG.md (select and tune on VAL only)")
    if a.golden == "check":
        return golden_check(reg, a.cached_only, a.fresh_only, a.test_id)
    if a.golden == "write":
        return golden_write(a.name or list(reg), reg, a.split, a.subset or None, a.fresh, a.tol)
    if len(a.name) != 1:
        sys.exit("one registry name (or --golden / --list)")
    over = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    parts = {}
    if a.route:
        parts["route"] = dict(head=a.route_head, tokens=a.route_tokens, step=a.route_step)
    if a.failure:
        parts["failure"] = dict(perts=a.failure.split(","), variant=a.variant, series_dir=a.series_dir, suffix=a.series_suffix, bs=a.series_bs, workers=a.series_workers)
    if a.timing:
        parts["timing"] = dict(n=a.timing_n, tag=f"_ev_{a.name[0]}")
    if parts and not a.tag:
        a.tag = "parts"  # a card with parts never replaces the standard card of the split (the experiments read that one)
    card = evaluate(a.name[0], reg, a.split, over, a.tau, a.subset or None, a.fresh, a.compare, a.tag, parts=parts or None)
    show(card)
    m = card["metrics"]
    bad = [f"{k} {m.get(k, float('nan')):.4f} != {v}" for k, v in (x.split("=") for x in a.expect) if not abs(m.get(k, float("nan")) - float(v)) <= TOL]
    if bad:
        sys.exit("stored numbers NOT reproduced: " + "; ".join(bad))
    if a.expect:
        print("stored numbers reproduced:", " ".join(a.expect))


if __name__ == "__main__":
    main()
