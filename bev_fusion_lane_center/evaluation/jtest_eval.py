"""The one-shot evaluation of the final lane system E312 on a SPLIT of the project (VAL for the check, TEST for the logged use of D84): B's prediction dump, the pair head's dump, the delivered graphs
(plain / connectors not snapped / connectors snapped, p >= 0.1), the topology numbers of the E312 cards, the maneuver coverage of E310 and the paired scene bootstrap of winner minus control.

    wsl bash tools/env/_py14.sh jtest_eval.py dump-b    --split val --seed 0 [--device cpu] [--out PATH] [--max N]
    wsl bash tools/env/_py14.sh jtest_eval.py dump-ends --split val --seed 0 [--run e312_p_s0] [--device cpu] [--out PATH] [--max N]
    wsl bash tools/env/_py14.sh jtest_eval.py score     --split val --seed 0 [--run e312_p_s0] [--b-dump PATH] [--ends-dump PATH] [--out DIR] [--boot 1000] [--max N]
    wsl bash tools/env/_py14.sh jtest_eval.py all       --split val --seeds 0 1 [--device cpu]        # dump-b, dump-ends, score per seed, then results.md (what run.py launches for the TEST use)
    wsl bash tools/env/_py14.sh jtest_eval.py verify    --seed 0 [--fresh]                            # VAL: reproduce the stored cards of E312 (ends_card_thr0.1.json, ends_card_snap_thr0.1.json, maneuver_val.json)
    wsl bash tools/env/_py14.sh jtest_eval.py selftest

Why a new file: `jends_eval.py`, `qends_eval.py` and `jmaneuver.py` score the 873 VAL frames only (the split, the dump paths and the card names are fixed in them); this module takes the split as a parameter and reuses their functions
(`qends_eval.add_connectors`, `qgraph_eval.readout / graph_counts / graph_numbers`, `jmaneuver.maneuvers / frame_counts`, `jends_eval.load_model`, `qgraph_readout.dump_full`) unchanged. With `--split val` it reproduces the stored VAL cards
(`verify`); the split switch is the only difference to what produced them.

THE TEST GATE (hard rule 1, D84). `--split test` is refused unless (1) `--test-id <run id>` is given, (2) the process runs under the launcher (`AR_RUN_ID` = that id, set by `autoresearch/tools/run.py`), and (3) the TEST log
(`evaluation/TEST_USAGE_LOG.md`) opens the split for the id by the gate that `evaluate.py` and `run.py` share (`autoresearch/tools/testgate.py`: a table row naming the id with the person's OK or a decision that grants a TEST check).
Nothing here selects, tunes or re-reads anything on the split: the read-outs (connectors at p >= 0.1, not snapped and snapped), the thresholds, the close-node step `close:1.5@0.5` and the metrics are the ones of the VAL cards.

Outputs: dumps (scratch, not in git) in `temp/e321/<split>/` (`predfull_e273_p_s<S>.pkl`, `ends_e312_p_s<S>.pkl`); cards in `runs/eval/e312_<split>/` (`card_s<S>.json`, `results.md`) for TEST, `temp/e321/val_check/` for VAL.
The graph GT of the frames is `graph_gt_val.pkl[row]` (that file, the tokens `val_tokens.json`, `poses_val.npz` and the features hold the VAL and the TEST rows of the cache; the rows of a split are `data/splits.json[split]`), the lane GT is the l6 store.
"""
import argparse
import json
import os
import pickle
import re
import sys
import time

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
sys.path.append(os.path.normpath(os.path.join(ROOT, "..", "autoresearch")))

SPLITS = {"val": "val_clean", "test": "test_clean"}
EXPECT_FRAMES = {"val": 873, "test": 833}
THR = 0.1                                                                                         # the delivery threshold of E309 / E312 (fixed on VAL)
CLOSE = "close:1.5@0.5"                                                                           # the named close-node step of E275 (fixed on VAL)
KEYS = ("Q", "F1@1.0", "lanes_found", "topo_F1", "topo_P", "topo_R", "junction_F1", "junction_P", "junction_R", "dup_frac", "vert100_dp")
LOWER = ("dup_frac", "vert100_dp")
VARIANTS = ("plain", "unsnapped", "snapped")                                                       # plain = B + the close-node step; unsnapped / snapped = + the connectors at p >= THR
MAN_NAME = {"plain": "plain", "unsnapped": "p0.1", "snapped": "p0.1s"}                              # the variant names of `jmaneuver.py`


# ------------------------------------------------------------------------------------------------------------------------------ the gate
def test_gate_open(test_id, env_run_id, log=None, decided=None):
    """(ok, why): the TEST split opens only for a logged id under the launcher; `log` / `decided` are for the tests (default: the project's log and its granting decisions, as evaluate.py)"""
    if not test_id:
        return False, "TEST refused: pass --test-id <run id> (the id of the run.py launch that the TEST log names)"
    if env_run_id != test_id:
        return False, f"TEST refused: the TEST split runs only under the launcher (`run.py --id {test_id} --split test -- ...`: AR_RUN_ID is '{env_run_id}', not '{test_id}')"
    import evaluate as EV                                                                            # the lane's evaluator: the same gate function, loaded only for TEST
    if not EV.test_allowed(test_id, log=log, decided=decided):
        return False, f"TEST refused: {test_id} is not opened by evaluation/TEST_USAGE_LOG.md (a table row naming the id with the person's OK)"
    return True, ""


def require_gate(a):
    if a.split == "test":
        ok, why = test_gate_open(a.test_id, os.environ.get("AR_RUN_ID", ""))
        if not ok:
            sys.exit(why)


# ------------------------------------------------------------------------------------------------------------------------------ frames
def split_rows(split):
    """the cache rows of a split (data/splits.json): ids only"""
    return np.asarray(json.load(open(f"{ROOT}/data/splits.json"))[split], int)


def frames_from_rows(rows):
    """Frame objects for cache rows: token (val_tokens.json), scene (poses_val.npz), the old graph GT (graph_gt_val.pkl); the files hold the VAL and the TEST rows of the cache, as `qgraph_eval.val_frames()` reads them"""
    import qgraph_eval as QE
    tokens = json.load(open(f"{QE.LS.B123}/feats/val_tokens.json"))
    scene = np.load(f"{QE.LS.B123}/poses_val.npz")["scene"]
    gt = pickle.load(open(f"{QE.LS.B123}/graph_gt_val.pkl", "rb"))
    assert rows.max() < min(len(tokens), len(scene), len(gt)), (rows.max(), len(tokens), len(scene), len(gt))
    return [QE.Frame(k, int(r), tokens[r], int(scene[r]), gt[r]) for k, r in enumerate(rows)]


def frames_of(split):
    """the frames of a split in the order of data/splits.json[split] (for val exactly `qgraph_eval.val_frames()`: the selftest checks it)"""
    rows = split_rows(split)
    assert len(rows) == EXPECT_FRAMES[split], (split, len(rows))
    return frames_from_rows(rows)


def paths(split, seed, a=None):
    d = os.path.join(ROOT, "temp", "e321", split)
    out = os.path.join(ROOT, "runs", "eval", f"e312_{split}") if split == "test" else os.path.join(ROOT, "temp", "e321", "val_check")
    return dict(dir=d, b=os.path.join(d, f"predfull_e273_p_s{seed}.pkl"), ends=os.path.join(d, f"ends_e312_p_s{seed}.pkl"), out=out)


# ------------------------------------------------------------------------------------------------------------------------------ dumps
def dump_b(split, seed, out, device="cpu", n=0):
    """B's full prediction pass on the split (exact map, as `qgraph_run_eval.score --dump-full`): the pickle `qgraph_readout.dump_full` writes"""
    import lane_gt_data as LD
    import qgraph_readout as RO
    import qpredict as QP
    import torch
    frames = frames_of(split)
    m = QP.load_model(os.path.join(ROOT, "runs", f"e273_p_s{seed}"), device)
    ds, dl = LD.loader(SPLITS[split], bs=4, workers=0, shuffle=False, map_mode="exact", pixels=False, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    assert rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's"
    preds, t0 = [], time.time()
    with torch.no_grad():
        for b in dl:
            preds += QP.predict(m, b)
            if n and len(preds) >= n:
                break
            if len(preds) % 100 < 4:
                print(f"  dump-b {len(preds)}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    preds = preds[: n or None]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    RO.dump_full(preds, rows[: len(preds)], out)
    print(f"dump-b {split} seed {seed}: {len(preds)} frames -> {out}  {time.time() - t0:.0f} s", flush=True)


def dump_ends(split, seed, run, out, device="cpu", n=0, thr=0.5):
    """the pair head's dump on the split: keypoints, pair logits, handle scales (as `jends_eval.dump`; the junction paint sidecars are not needed: the model reads the features only, the GT ends come from the lane GT)"""
    import jends_eval as JE
    import lane_gt_data as LD
    import qendnet as QEN
    import torch
    frames = frames_of(split)
    m, args = JE.load_model(run, device)
    ds, dl = LD.loader(SPLITS[split], bs=8, workers=0, shuffle=False, map_mode="error", pixels=False, ends=True, strict=True)
    rows = [int(ds.bd.rows[k]) for k in ds.ks]
    assert rows == [f.row for f in frames], "the frame order of the loader differs from the scorer's"
    out_frames, tot, t0 = [], {}, time.time()
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device).float()
            _, _, oe = m(x, gt=None)
            for k, v in QEN.end_counts(oe, b, thr).items():
                tot[k] = tot.get(k, 0) + v
            for i in range(len(b["token"])):
                out_frames.append(dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=oe["kp_pos"][i].cpu().numpy(), kp_dir=oe["kp_dir"][i].cpu().numpy(), kp_p=oe["kp_p"][i].cpu().numpy(),
                                       valid=oe["valid"][i].cpu().numpy(), logit=oe["logit"][i].float().cpu().numpy(), allowed=oe["allowed"][i].cpu().numpy(),
                                       scale=oe["scale"][i].float().cpu().numpy() if "scale" in oe else None))
            if len(out_frames) % 80 == 0:
                print(f"  dump-ends {len(out_frames)}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
            if n and len(out_frames) >= n:
                break
    out_frames = out_frames[: n or None]
    r = lambda x, y: x / max(y, 1)  # noqa: E731
    head = dict(ent_recall=r(tot["ent_hit"], tot["ent_n"]), ent_precision=r(tot["ent_ok"], tot["ent_kp"]), ext_recall=r(tot["ext_hit"], tot["ext_n"]), ext_precision=r(tot["ext_ok"], tot["ext_kp"]),
                pair_recall=r(tot["pairs_hit"], tot["pairs_n"]), pair_precision=r(tot["pairs_ok"], tot["pairs_kp"]), counts=tot, thr=thr, frames=len(out_frames), run=run)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump(out_frames, open(out, "wb"))
    json.dump(head, open(out.replace(".pkl", "_head.json"), "w"), indent=1)
    print(f"dump-ends {split} seed {seed}: {len(out_frames)} frames -> {out}  {time.time() - t0:.0f} s; ent recall {head['ent_recall']:.3f} exit recall {head['ext_recall']:.3f} pair recall {head['pair_recall']:.3f}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------ scoring
def pairs_of(e, thr, chord=False):
    """the pairs of the head at probability >= thr: (entry xy, entry heading, exit xy, exit heading[, the two handle scales]) as `qends_eval.cmd_score`"""
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= thr)
    return [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None and not chord else ()) for i, j in zip(ia, ib)]


def score_dumps(frames, dump_b_, ends_, thr=THR, n=0, log=print):
    """per frame: the counts of the three delivered graphs and the maneuver counts -> dict(counts={variant: [per-frame counts]}, man={variant: [(realised, maneuvers, hit, near)]}, curves, wired)"""
    import jmaneuver as JM
    import qends as QEND
    import qends_eval as QEV
    import qgraph_eval as QE
    import qgraph_readout as QR
    assert [d["row"] for d in dump_b_] == [f.row for f in frames][: len(dump_b_)], "B's full dump is not in the order of the frames"
    spec = QR.parse(CLOSE)
    cs = {v: [] for v in VARIANTS}
    man = {v: [] for v in VARIANTS}
    curves = {v: 0 for v in VARIANTS[1:]}
    wired = {v: 0 for v in VARIANTS[1:]}
    t0 = time.time()
    for b, (f, d) in enumerate(zip(frames, dump_b_)):
        if n and b >= n:
            break
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QE.readout(pred, 0.5, 0.5)
        long_mask = np.asarray(pred["links"]["long"], bool)
        e = ends_[f.row]
        pr = pairs_of(e, thr)
        raws = {"plain": QE.to_raw(graph)}
        for v, snap in (("unsnapped", False), ("snapped", True)):
            raws[v], nc, nw = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, pr, snap=snap)
            curves[v] += nc
            wired[v] += nw
        cs["plain"].append(QE.graph_counts(graph, f))
        cs["unsnapped"].append(QE.graph_counts(raws["unsnapped"], f))
        cs["snapped"].append(QE.graph_counts(raws["snapped"], f))
        g = f.lane()
        mans = JM.maneuvers(g, QEND.cluster)
        if mans:
            ref_tree = f.ref().tree
            for v in VARIANTS:
                n_ok, nm, hit, near, _ = JM.frame_counts(raws[v], mans, ref_tree)
                man[v].append((n_ok, nm, hit, near))
        f._lane = f._ref = None
        if (b + 1) % 100 == 0:
            log(f"  scored {b + 1}/{len(frames)}  {time.time() - t0:.0f} s")
    return dict(counts=cs, man=man, curves=curves, wired=wired)


def maneuver_numbers(man):
    out = {}
    for v, rows in man.items():
        x = np.array(rows)
        out[MAN_NAME[v]] = dict(maneuvers=int(x[:, 1].sum()), realised=int(x[:, 0].sum()), recall=float(x[:, 0].sum() / x[:, 1].sum()), frame_recall_mean=float(np.mean(x[:, 0] / x[:, 1])), frames_none=int((x[:, 0] == 0).sum()),
                                frames_all=int((x[:, 0] == x[:, 1]).sum()), junction_precision=float(x[:, 2].sum() / max(1, x[:, 3].sum())), frames=len(x))
    return out


def paired(cs_a, cs_b, scenes, boot=1000, keys=KEYS):
    """winner minus control by the project's paired scene bootstrap (`autoresearch/Evaluation/stats.py`: a = variant, b = control; whole scenes drawn with replacement, the same draws for both)"""
    import qgraph_eval as QE
    from Evaluation import stats as ST
    f_a = lambda sel: QE.graph_numbers(cs_a if sel is None else [cs_a[i] for i in sel])  # noqa: E731
    f_b = lambda sel: QE.graph_numbers(cs_b if sel is None else [cs_b[i] for i in sel])  # noqa: E731
    return ST.paired_draws(f_a, f_b, ST.scene_draws(scenes, boot, 0), keys, better={k: "low" for k in LOWER})


def make_card(a_seed, run, split, frames, sc, boot):
    import qgraph_eval as QE
    from qends_eval import holds
    num = {v: QE.graph_numbers(sc["counts"][v]) for v in VARIANTS}
    card = dict(split=split, run=run, seed=a_seed, frames=len(frames), scenes=int(len(set(f.scene for f in frames))), thr=THR, close_step=CLOSE, date=time.strftime("%Y-%m-%d %H:%M"), variants={})
    for v in VARIANTS[1:]:
        ok, cond = holds(num[v], num["plain"])
        card["variants"][v] = dict(curves=sc["curves"][v], wired_both_ends=sc["wired"][v], numbers=dict(base={k: float(num["plain"][k]) for k in KEYS}, head={k: float(num[v][k]) for k in KEYS}),
                                   difference={k: float(num[v][k] - num["plain"][k]) for k in KEYS}, rule_d79_this_seed=dict(holds=bool(ok), conditions={k: bool(x) for k, x in cond.items()}))
    card["maneuver"] = maneuver_numbers(sc["man"])
    scenes = [f.scene for f in frames][: len(sc["counts"]["plain"])]
    if boot:
        card["paired"] = {v: paired(sc["counts"][v], sc["counts"]["plain"], scenes, boot) for v in VARIANTS[1:]}
        card["paired_note"] = f"winner minus control (the plain delivered graph of B + the close-node step), {boot} scene draws of {len(set(scenes))} scenes, same draws for both (autoresearch/Evaluation/stats.py)"
    return card


# ------------------------------------------------------------------------------------------------------------------------------ commands
def cmd_dump_b(a):
    require_gate(a)
    dump_b(a.split, a.seed, a.out or paths(a.split, a.seed)["b"], a.device, a.max)


def cmd_dump_ends(a):
    require_gate(a)
    dump_ends(a.split, a.seed, a.run or f"e312_p_s{a.seed}", a.out or paths(a.split, a.seed)["ends"], a.device, a.max)


def cmd_score(a):
    require_gate(a)
    p = paths(a.split, a.seed)
    frames = frames_of(a.split)
    dump_b_ = pickle.load(open(a.b_dump or p["b"], "rb"))
    ends_ = {f["row"]: f for f in pickle.load(open(a.ends_dump or p["ends"], "rb"))}
    run = a.run or f"e312_p_s{a.seed}"
    ck = json.load(open(os.path.join(ROOT, "runs", run, "config.json")))
    assert ck.get("seed", a.seed) == a.seed, f"--seed {a.seed} does not match the run {run} (seed {ck.get('seed')})"
    sc = score_dumps(frames, dump_b_, ends_, n=a.max)
    card = make_card(a.seed, run, a.split, frames[: len(sc['counts']['plain'])], sc, a.boot)
    out = a.out or p["out"]
    os.makedirs(out, exist_ok=True)
    json.dump(card, open(os.path.join(out, f"card_s{a.seed}.json"), "w"), indent=1)
    print_card(card)


def print_card(card):
    print(f"{card['split']} seed {card['seed']} run {card['run']}: {card['frames']} frames, {card['scenes']} scenes")
    print(f"{'':20s}" + "".join(f"{k:>12s}" for k in KEYS))
    for v in VARIANTS[1:]:
        c = card["variants"][v]
        if v == VARIANTS[1]:
            print(f"{'plain graph':20s}" + "".join(f"{c['numbers']['base'][k]:12.4f}" for k in KEYS))
        print(f"{v:20s}" + "".join(f"{c['numbers']['head'][k]:12.4f}" for k in KEYS))
        print(f"{'  - plain':20s}" + "".join(f"{c['difference'][k]:+12.4f}" for k in KEYS))
    for v, m in card["maneuver"].items():
        print(f"maneuvers {v:6s} realised {m['realised']}/{m['maneuvers']} = {m['recall']:.3f}; junction precision {m['junction_precision']:.3f}")
    for v, p in card.get("paired", {}).items():
        print(f"paired {v} - plain: " + ", ".join(f"{k} {p[k]['diff']:+.4f} [{p[k]['lo']:+.4f}, {p[k]['hi']:+.4f}]" for k in ("topo_F1", "lanes_found", "junction_F1", "dup_frac") if k in p))


def results_md(cards, split):
    L = [f"# E312 on {split.upper()} ({cards[0]['frames']} frames, {cards[0]['scenes']} scenes) - winner against the plain delivered graph of B", "",
         "Read-outs fixed on VAL: connectors at p >= 0.1 snapped (E311, the card of record of E312) and not snapped (the differentiable read-out); control = B + the close-node step `close:1.5@0.5`. Report only: nothing selected or tuned on this split.", ""]
    for v in VARIANTS[1:]:
        L += [f"## {v} (connectors at p >= {THR})", "", "| seed | topo F1 plain | topo F1 | diff [95 % scene bootstrap] | lanes found diff | junction F1 diff | dup_frac diff | F1@1.0 diff | vertices / 100 m diff |", "|---|---|---|---|---|---|---|---|---|"]
        for c in cards:
            d, h, p = c["variants"][v]["difference"], c["variants"][v]["numbers"], c.get("paired", {}).get(v, {})
            ci = lambda k: f"{d[k]:+.4f}" + (f" [{p[k]['lo']:+.4f}, {p[k]['hi']:+.4f}]" if k in p else "")  # noqa: E731
            L.append(f"| {c['seed']} | {h['base']['topo_F1']:.4f} | {h['head']['topo_F1']:.4f} | {ci('topo_F1')} | {ci('lanes_found')} | {ci('junction_F1')} | {ci('dup_frac')} | {ci('F1@1.0')} | {ci('vert100_dp')} |")
        L.append("")
    L += ["## maneuver coverage (E310)", "", "| seed | plain | connectors not snapped | connectors snapped |", "|---|---|---|---|"]
    for c in cards:
        m = c["maneuver"]
        L.append(f"| {c['seed']} | {100 * m['plain']['recall']:.1f} % | {100 * m['p0.1']['recall']:.1f} % | {100 * m['p0.1s']['recall']:.1f} % |")
    return "\n".join(L) + "\n"


def cmd_all(a):
    require_gate(a)
    cards = []
    for s in a.seeds:
        p = paths(a.split, s)
        t0 = time.time()
        dump_b(a.split, s, p["b"], a.device, a.max)
        dump_ends(a.split, s, f"e312_p_s{s}", p["ends"], a.device, a.max)
        frames = frames_of(a.split)
        dump_b_ = pickle.load(open(p["b"], "rb"))
        ends_ = {f["row"]: f for f in pickle.load(open(p["ends"], "rb"))}
        sc = score_dumps(frames, dump_b_, ends_, n=a.max)
        card = make_card(s, f"e312_p_s{s}", a.split, frames[: len(sc["counts"]["plain"])], sc, a.boot)
        os.makedirs(p["out"], exist_ok=True)
        json.dump(card, open(os.path.join(p["out"], f"card_s{s}.json"), "w"), indent=1)
        print_card(card)
        cards.append(card)
        print(f"seed {s} done in {time.time() - t0:.0f} s", flush=True)
    open(os.path.join(paths(a.split, a.seeds[0])["out"], "results.md"), "w").write(results_md(cards, a.split))


def _diff(new, old, path=""):
    """max abs difference of the numbers present in both (nested dicts), with where it sits"""
    worst = (0.0, "")
    for k, v in old.items():
        if k not in new:
            continue
        if isinstance(v, dict):
            w = _diff(new[k], v, f"{path}/{k}")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            w = (abs(float(new[k]) - float(v)), f"{path}/{k}")
        else:
            continue
        if w[0] > worst[0]:
            worst = w
    return worst


def cmd_verify(a):
    """VAL: the scorer of this module against the stored cards of E312 (both seeds if --seed is not given); with --fresh the dumps are made again on this device first (the check of the dump code, CPU against the GPU that made the stored ones)"""
    seed = a.seed
    run = f"e312_p_s{seed}"
    p = paths("val", seed)
    stored = dict(snapped=json.load(open(os.path.join(ROOT, "runs", run, "ends_card_snap_thr0.1.json"))), unsnapped=json.load(open(os.path.join(ROOT, "runs", run, "ends_card_thr0.1.json"))),
                  man=json.load(open(os.path.join(ROOT, "runs", run, "maneuver_val.json"))))
    if a.fresh:
        dump_b("val", seed, p["b"], a.device, a.max)
        dump_ends("val", seed, run, p["ends"], a.device, a.max)
        b_path, e_path = p["b"], p["ends"]
    else:
        b_path, e_path = os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), os.path.join(ROOT, "runs", run, "ends_val.pkl")
    frames = frames_of("val")
    dump_b_ = pickle.load(open(b_path, "rb"))
    ends_ = {f["row"]: f for f in pickle.load(open(e_path, "rb"))}
    if a.fresh:
        sb, se = pickle.load(open(os.path.join(ROOT, "temp", "e276", f"predfull_e273_p_s{seed}.pkl"), "rb")), pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
        n = len(dump_b_)
        print("dump-b against the stored dump: max abs diff of pos / e / link_p over the frames:", max(float(np.abs(np.asarray(x["pos"]) - np.asarray(y["pos"])).max()) if len(x["pos"]) == len(y["pos"]) else float("inf") for x, y in zip(dump_b_, sb[:n])),
              max(float(np.abs(np.asarray(x["e"]) - np.asarray(y["e"])).max()) if len(x["e"]) == len(y["e"]) else float("inf") for x, y in zip(dump_b_, sb[:n])),
              "nodes differ in count in", sum(len(x["e"]) != len(y["e"]) for x, y in zip(dump_b_, sb[:n])), "frames")
        print("dump-ends against the stored dump: max abs diff of kp_pos / logit / scale:", max(float(np.abs(x["kp_pos"] - y["kp_pos"]).max()) for x, y in zip(pickle.load(open(e_path, "rb")), se[:n])),
              max(float(np.abs(x["logit"] - y["logit"]).max()) for x, y in zip(pickle.load(open(e_path, "rb")), se[:n])), max(float(np.abs(x["scale"] - y["scale"]).max()) for x, y in zip(pickle.load(open(e_path, "rb")), se[:n])))
    sc = score_dumps(frames, dump_b_, ends_, n=a.max)
    card = make_card(seed, run, "val", frames[: len(sc["counts"]["plain"])], sc, a.boot)
    os.makedirs(p["out"], exist_ok=True)
    json.dump(card, open(os.path.join(p["out"], f"card_s{seed}{'_fresh' if a.fresh else ''}.json"), "w"), indent=1)
    print_card(card)
    ok = True
    for v, key in (("snapped", "snapped"), ("unsnapped", "unsnapped")):
        new, old = card["variants"][v], stored[key]
        d_head, d_base = _diff(new["numbers"], old["numbers"]), _diff(new["numbers"]["base"], old["numbers"]["base"])
        d_diff = _diff(new["difference"], old["difference"])
        same = (new["curves"] == old["curves"] and new["wired_both_ends"] == old["wired_both_ends"])
        print(f"VERIFY {v}: max abs difference of the numbers {d_head[0]:.2e} ({d_head[1]}), of the plain numbers {d_base[0]:.2e}, of the differences {d_diff[0]:.2e}; curves {new['curves']} / {old['curves']}, wired {new['wired_both_ends']} / {old['wired_both_ends']} ({'equal' if same else 'DIFFER'})")
        ok = ok and d_head[0] < 1e-6 and same
    for v in ("plain", "p0.1", "p0.1s"):
        new, old = card["maneuver"][v], stored["man"]["variants"][v]
        d = _diff(new, old)
        print(f"VERIFY maneuver {v}: realised {new['realised']} / {old['realised']} of {new['maneuvers']} / {old['maneuvers']}, max abs difference {d[0]:.2e} ({d[1]})")
        ok = ok and d[0] < 1e-6
    print("VERIFY RESULT:", "all reproduced (to 1e-6)" if ok else "DIFFERENCES (see above)")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------------------ self-test
def selftest():
    import tempfile
    # (1) the gate: refuses TEST without an id, outside the launcher, with an id the log does not open; opens only for a logged id under the launcher
    log = "| date | loop | what | who / which run | numbers | note |\n|---|---|---|---|---|---|\n| d | 1 | design | run id E9.ok | - | Mirko's OK, once |\n| d | 1 | design | run id E9.no | - | pending |\n"
    assert not test_gate_open("", "", log, set())[0]
    assert not test_gate_open("E9.ok", "", log, set())[0], "outside the launcher"
    assert not test_gate_open("E9.ok", "E9.other", log, set())[0], "AR_RUN_ID of another run"
    assert not test_gate_open("E9.no", "E9.no", log, set())[0], "a row without the OK"
    assert not test_gate_open("E9.none", "E9.none", log, set())[0], "an id the log does not name"
    assert test_gate_open("E9.ok", "E9.ok", log, set())[0]
    a = argparse.Namespace(split="test", test_id="E9.ok")
    old = os.environ.pop("AR_RUN_ID", None)
    try:
        try:
            require_gate(a)
            raise AssertionError("require_gate let TEST through without the launcher")
        except SystemExit:
            pass
        a.split = "val"
        require_gate(a)                                                                            # VAL needs no gate
    finally:
        if old is not None:
            os.environ["AR_RUN_ID"] = old
    # (2) the splits: ids only
    r_val, r_test = split_rows("val"), split_rows("test")
    assert len(r_val) == EXPECT_FRAMES["val"] and len(r_test) == EXPECT_FRAMES["test"] and not set(r_val.tolist()) & set(r_test.tolist())
    # (3) the paired bootstrap on toy counts (graph_numbers replaced by a mean of one field): a variant that is 0.1 better in every frame gives an interval around +0.1, a system against itself exactly 0
    import qgraph_eval as QE
    saved = QE.graph_numbers
    QE.graph_numbers = lambda cs: {"topo_F1": float(np.mean([c["x"] for c in cs])), "dup_frac": float(np.mean([c["d"] for c in cs]))}
    try:
        rng = np.random.RandomState(1)
        base_c = [dict(x=float(v), d=0.2) for v in rng.rand(40)]
        var_c = [dict(x=c["x"] + 0.1, d=0.2) for c in base_c]
        scenes = [i // 5 for i in range(40)]
        p = paired(var_c, base_c, scenes, 200, keys=("topo_F1", "dup_frac"))
        assert abs(p["topo_F1"]["diff"] - 0.1) < 1e-9 and abs(p["topo_F1"]["lo"] - 0.1) < 1e-9 and abs(p["topo_F1"]["hi"] - 0.1) < 1e-9, p["topo_F1"]
        p0 = paired(base_c, base_c, scenes, 200, keys=("topo_F1",))
        assert p0["topo_F1"]["diff"] == 0.0 and p0["topo_F1"]["lo"] == 0.0 and p0["topo_F1"]["hi"] == 0.0
    finally:
        QE.graph_numbers = saved
    # (4) the frames of the split val are exactly those of the scorers (`qgraph_eval.val_frames()`); the test branch is the same code on the test rows
    fv, fo = frames_of("val"), QE.val_frames()
    assert [(f.k, f.row, f.token, f.scene) for f in fv] == [(f.k, f.row, f.token, f.scene) for f in fo] and all(len(x.gt["pts"]) == len(y.gt["pts"]) for x, y in zip(fv, fo))
    # (5) _diff and the results text
    assert _diff(dict(a=1.0, b=dict(c=2.0)), dict(a=1.0, b=dict(c=2.5)))[0] == 0.5
    card = dict(split="val", seed=0, frames=3, scenes=2, run="r", variants={v: dict(difference={k: 0.01 for k in KEYS}, numbers=dict(base={k: 0.5 for k in KEYS}, head={k: 0.51 for k in KEYS})) for v in VARIANTS[1:]},
                maneuver={"plain": dict(recall=0.5), "p0.1": dict(recall=0.7), "p0.1s": dict(recall=0.71)})
    txt = results_md([card], "val")
    assert "snapped" in txt and "maneuver coverage" in txt
    print("jtest_eval selftest ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s, seeds=False):
        s.add_argument("--split", required=True, choices=("val", "test"))
        s.add_argument("--test-id", default="", help="the run id that the TEST log names (--split test only; the process must run under run.py with that id)")
        s.add_argument("--device", default="cpu")
        s.add_argument("--max", type=int, default=0, help="the first N frames only (a smoke test)")
        if seeds:
            s.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
            s.add_argument("--boot", type=int, default=1000)
        else:
            s.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("dump-b")
    common(s)
    s.add_argument("--out", default=None)
    s = sub.add_parser("dump-ends")
    common(s)
    s.add_argument("--run", default=None)
    s.add_argument("--out", default=None)
    s = sub.add_parser("score")
    common(s)
    s.add_argument("--run", default=None)
    s.add_argument("--b-dump", default=None)
    s.add_argument("--ends-dump", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--boot", type=int, default=1000)
    s = sub.add_parser("all")
    common(s, seeds=True)
    s = sub.add_parser("verify")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--fresh", action="store_true")
    s.add_argument("--device", default="cpu")
    s.add_argument("--max", type=int, default=0)
    s.add_argument("--boot", type=int, default=0)
    sub.add_parser("selftest")
    a = ap.parse_args()
    for attr in ("out", "b_dump", "ends_dump"):                                                    # the script runs in evaluation/ (the launcher's wrapper changes into its folder): paths are relative to the project root
        v = getattr(a, attr, None)
        if v and not os.path.isabs(v):
            setattr(a, attr, os.path.join(ROOT, v))
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "verify":
        sys.exit(cmd_verify(a))
    {"dump-b": cmd_dump_b, "dump-ends": cmd_dump_ends, "score": cmd_score, "all": cmd_all}[a.cmd](a)


if __name__ == "__main__":
    main()
