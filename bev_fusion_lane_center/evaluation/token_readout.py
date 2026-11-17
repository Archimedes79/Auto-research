"""Read-out helpers of the token heads (E365 family: S2 E368, S3 E374, ...; promoted on 2026-11-17 from the scratch scripts `choose.py`, `ap.py`, `mk_every3.py`, `modes.py` the cards E368 / E374 name as how their numbers were made). They only READ the dumps and sweep tables
that `jtoken_eval.py` (and `jpast_eval.py` / `jmap_eval.py`, which write the same format) leave in `runs/<run>/`; no GPU, no GT field, nothing written but the every-3 file.

    wsl bash tools/env/_py14.sh token_readout.py choose RUN [RUN ...]                    # the cell chosen on the run's OWN TRAIN sample sweep table (runs/RUN/token_sweep_train_every3.json) by the E366 rule -> 'link keep gate'
    wsl bash tools/env/_py14.sh token_readout.py every3 RUN [RUN ...] [--ntrain 8414]    # runs/RUN/token_train_every3.pkl = every third frame of the 873-frame TRAIN dump (what a `--split train` dump of jpast_eval / jmap_eval leaves; the sweep reads this file)
    wsl bash tools/env/_py14.sh token_readout.py ap RUN [RUN ...]                        # existence AP on the VAL record dump: 1 - p_none of the valid tokens against 'assigned to a GT node' (threshold-free)
    wsl bash tools/env/_py14.sh token_readout.py modes SEED M_RUN Z_RUN LINK,KEEP,GATE [MODE ...]   # the control dumps runs/M_RUN_<mode>/ (default clean gross none S U C2) scored at ONE cell next to the record dump of M_RUN and the control arm Z_RUN,
                                                                                         # each mode with its paired scene-bootstrap interval against the record (err) and against Z
    wsl bash tools/env/_py14.sh token_readout.py selftest                                # the hand cases of `choose_cell`, `existence_ap`, `every3` (tests/test_token_readout.py adds the mutants)

E366 rule (`choose_cell`): among the cells of the table with dup_frac <= 0.05 the highest topo F1; exact ties go to the cell nearest the decode of record (link threshold, then keep threshold, then the none gate). The cell is chosen on the TRAIN sample only and applied once to VAL.
Paired intervals: `jtoken_eval.QE.paired` (scene bootstrap, 1000 draws). Another scorer of the same dumps: `jtoken_eval.py score` (`--ref-run` for a paired interval against another run).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training/data"):
    sys.path.insert(0, os.path.join(ROOT, sub_))
REC = (0.3, 0.5, True)                    # the decode of record: link threshold, keep threshold, none gate on
MAX_DUP = 0.05                            # the duplicate guard of the E366 rule
NTRAIN = 8414                             # TRAIN frames of the E365 cache (the every-3 sample is taken from them)


def choose_cell(rows, rec=REC, max_dup=MAX_DUP):
    """rows of a sweep table (dicts topo_F1, link_thr, keep_thr, none_gate, dup_frac) -> (topo F1, (link, keep, gate), dup_frac) of the cell chosen by the E366 rule (module docstring)"""
    ok = [(r["topo_F1"], (r["link_thr"], r["keep_thr"], bool(r["none_gate"])), r["dup_frac"]) for r in rows if r["dup_frac"] <= max_dup]
    assert ok, f"no cell with dup_frac <= {max_dup}"
    best = max(r[0] for r in ok)
    dist = lambda c: (abs(c[0] - rec[0]), abs(c[1] - rec[1]), c[2] != rec[2])  # noqa: E731
    return sorted([r for r in ok if abs(r[0] - best) < 1e-12], key=lambda r: dist(r[1]))[0]


def existence_ap(scores, y):
    """average precision of `scores` for the boolean labels `y` (sklearn; ties as sklearn takes them)"""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y, bool), np.asarray(scores, float)))


def every3(meta, frames, ntrain, sample):
    """the every-third subsample of a TRAIN dump (meta, frames): every per-frame list of the meta (and of meta['use']) is cut with [::3], meta['every'] = 3; the sample must be the read-out sample of `jtoken_eval.sample(ntrain, 'train', 3)` -> (meta, frames)"""
    n = len(meta["idx"])
    assert n == len(frames), (n, len(frames))
    m3 = dict(meta)
    for k, v in meta.items():
        if isinstance(v, list) and len(v) == n and k != "use":
            m3[k] = v[::3]
    if isinstance(meta.get("use"), dict):
        m3["use"] = {k: (v[::3] if isinstance(v, list) and len(v) == n else v) for k, v in meta["use"].items()}
    m3["every"] = 3
    assert m3["idx"] == sample(ntrain, "train", 3).tolist(), "the every-3 sample of the dump is not the one of jtoken_eval.sample"
    return m3, frames[::3]


# ------------------------------------------------------------------------------------------------------------------------------ commands
def cmd_choose(runs):
    for run in runs:
        t = json.load(open(os.path.join(ROOT, "runs", run, "token_sweep_train_every3.json")))
        top, cell, dup = choose_cell(t["rows"])
        print(run, "cell", cell[0], cell[1], "gate_on" if cell[2] else "gate_off", "| TRAIN topo F1 %.4f dup %.4f | frames %s" % (top, dup, t["frames"]))


def cmd_every3(runs, ntrain):
    import jtoken_eval as JE
    for run in runs:
        src, dst = (os.path.join(ROOT, "runs", run, f) for f in ("token_train.pkl", "token_train_every3.pkl"))
        assert os.path.exists(src) and not os.path.exists(dst), (src, dst)
        with open(src, "rb") as fh:
            d = pickle.load(fh)
        meta, fr = every3(d["meta"], d["frames"], ntrain, JE.sample)
        with open(dst, "wb") as fh:
            pickle.dump(dict(d, meta=meta, frames=fr), fh, protocol=4)
        print(run, "->", dst, len(fr), "frames")


def cmd_ap(runs):
    import torch
    import lane_ptok_data as LP
    import lane_tokcache as LC
    for run in runs:
        seed = int(torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu", weights_only=False)["args"]["seed"])
        with open(os.path.join(ROOT, "runs", run, "token_val.pkl"), "rb") as fh:
            d = pickle.load(fh)
        c = LC.TokCache(LP.cache_dir("val", seed, "e292_a"))
        tok2i = {t: i for i, t in enumerate(c.tokens)}
        s, y = [], []
        for f in d["frames"]:
            i = tok2i[f["token"]]
            v = np.array(c.f["valid"][i])
            s.append(1.0 - np.asarray(f["pnone"], float)[v])
            y.append(np.array(c.f["gt_id"][i])[v] >= 0)
        s, y = np.concatenate(s), np.concatenate(y)
        print("%-14s tokens %d assigned share %.3f | AP of the existence %.4f" % (run, len(y), y.mean(), existence_ap(s, y)))


def cmd_modes(seed, m_run, z_run, cell, modes):
    import jtoken_eval as JE
    lt, kt, g = cell.split(",")

    def load(run):
        a = argparse.Namespace(run=run, model="model.pt", seed=seed, split="val", every=1, drop=0.0, drop_seed=0, max=0)
        r = JE.read_dump(a, JE.RUNS, 0)
        fr = [JE.decode_variant(f, v, float(lt), float(kt), g == "on", None) for f, v in zip(r["fr"], r["valid"])]
        return JE.score_preds([JE.frame_prediction(t, f, "rec", float(kt)) for t, f in zip(r["toks"], fr)], r["frames"], log=lambda *x, **k: None), r["frames"]

    rec, frames = load(m_run)
    ctl, _ = load(z_run)
    fmt = lambda r: "%.4f %.4f %.4f %.4f" % tuple(r["graph"][k] for k in ("topo_F1", "lanes_found", "junction_F1", "dup_frac"))  # noqa: E731
    d = lambda q: {x: round(v, 4) for x, v in q["topo_F1"].items() if x in ("diff", "lo", "hi")}  # noqa: E731
    print(f"seed {seed}, cell {cell}, VAL {len(frames)} frames: topo_F1 / lanes_found / junction_F1 / dup_frac")
    print("  %-8s %s" % ("z", fmt(ctl)))
    print("  %-8s %s" % ("err", fmt(rec)))
    for m in modes:
        r, _ = load(f"{m_run}_{m}")
        print("  %-8s %s | vs err topo_F1 %s | vs z %s" % (m, fmt(r), d(JE.QE.paired(r["rec"], rec["rec"], frames)), d(JE.QE.paired(r["rec"], ctl["rec"], frames))))


# ------------------------------------------------------------------------------------------------------------------------------ selftest
def selftest():
    row = lambda t, l, k, g, d: dict(topo_F1=t, link_thr=l, keep_thr=k, none_gate=g, dup_frac=d)  # noqa: E731
    out, a, off, on, low = row(0.70, 0.1, 0.6, False, 0.06), row(0.69, 0.2, 0.6, False, 0.05), row(0.69, 0.3, 0.5, False, 0.04), row(0.69, 0.3, 0.5, True, 0.04), row(0.60, 0.3, 0.5, True, 0.01)
    top, cell, dup = choose_cell([out, a, off, on, low])
    assert top == 0.69 and cell == (0.3, 0.5, True) and dup == 0.04, "the 0.70 cell has dup 0.06 > 0.05 (out); of the three exact ties the one with the record's thresholds AND gate wins, whatever the order of the table"
    assert choose_cell([out, a, off])[1] == (0.3, 0.5, False), "a tie between (0.2, 0.6, off) and (0.3, 0.5, off): the one with the record's thresholds (distance 0) wins"
    assert choose_cell([row(0.69, 0.2, 0.5, False, 0.04), row(0.69, 0.3, 0.7, False, 0.04)])[1] == (0.3, 0.7, False), "the link threshold is compared first (distance 0 against 0.1 beats the keep threshold's 0.2 against 0)"
    assert choose_cell([row(0.5, 0.1, 0.5, True, 0.05)])[1] == (0.1, 0.5, True), "dup_frac == 0.05 is allowed"
    try:
        choose_cell([row(0.5, 0.1, 0.5, True, 0.051)])
        raise RuntimeError("no cell inside the guard must be refused")
    except AssertionError:
        pass
    assert abs(existence_ap([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) - (1 + 2 / 3) / 2) < 1e-12 and existence_ap([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0, "AP by hand: precisions 1 and 2/3 at the two positives"
    assert existence_ap([0.1, 0.2, 0.9], [1, 1, 0]) < 0.6, "inverted scores score low"
    ntrain = 9000
    sample = lambda n, split, every: np.arange(0, n, 10)[::every]  # noqa: E731    (a stand-in sample: 900 frames, every third of them 300)
    idx = np.arange(0, ntrain, 10)
    meta = dict(idx=idx.tolist(), run="r", drop=0.0, use=dict(v=list(range(900)), total=5.0))
    frames = [dict(token=f"t{i}") for i in range(900)]
    m3, f3 = every3(meta, frames, ntrain, sample)
    assert m3["every"] == 3 and m3["idx"] == idx[::3].tolist() and len(f3) == 300 and f3[1]["token"] == "t3" and m3["use"]["v"] == list(range(0, 900, 3)) and m3["use"]["total"] == 5.0 and m3["run"] == "r" and meta["idx"] == idx.tolist(), "every third frame, the per-frame lists cut, the rest kept, the input unchanged"
    try:
        every3(dict(meta, idx=idx[::2].tolist()), frames[:450], ntrain, sample)
        raise RuntimeError("another sample must be refused")
    except AssertionError:
        pass
    print("token_readout selftest ok")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    sub = ap_.add_subparsers(dest="cmd", required=True)
    for name in ("choose", "ap"):
        sub.add_parser(name).add_argument("runs", nargs="+")
    s = sub.add_parser("every3")
    s.add_argument("runs", nargs="+")
    s.add_argument("--ntrain", type=int, default=NTRAIN)
    s = sub.add_parser("modes")
    s.add_argument("seed", type=int)
    s.add_argument("m_run")
    s.add_argument("z_run")
    s.add_argument("cell", help="link,keep,gate (on|off), e.g. 0.1,0.6,off")
    s.add_argument("modes", nargs="*", default=["clean", "gross", "none", "S", "U", "C2"])
    sub.add_parser("selftest")
    a = ap_.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "choose":
        cmd_choose(a.runs)
    elif a.cmd == "every3":
        cmd_every3(a.runs, a.ntrain)
    elif a.cmd == "ap":
        cmd_ap(a.runs)
    else:
        cmd_modes(a.seed, a.m_run, a.z_run, a.cell, a.modes)
