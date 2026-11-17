"""E358 timing of the WIRING STAGE (scratch, CPU, host side; the 'faster?' fact of D106 for the learned wiring head): per VAL frame of the end-to-end read-out of E324 s (attached ends, snap wiring without the snap geometry), the time from the pairs to the delivered curves with
 A = the nearest-piece distance rule  (`add_connectors(..., wire_choice=None)`: the rule's own scan of the piece ends inside),
 B = the learned head              (`jwire.kept_ends` + `jwire_head.piece_attrs` (the candidate lists and their attributes) + `wire_choice_head` (the head's forward) + `add_connectors(..., wire_choice=wc)`),
and B split into its three parts. Not timed: the read-out of B's graph (`QR.apply`, `QE.readout`), `make_pairs`, the loading. Protocol (the A/B timer of jtiming, in short): all frames' inputs are prepared once; a ROUND times every frame for A and B, the order alternates by round (A B | B A | A B)
and by frame parity inside a round, so a drift of the host cancels in the paired differences; a frame's number is the median over the rounds; the difference B - A is the mean over frames of the per-frame paired differences with a 95 % interval of a bootstrap over frames (1000 resamples). One process, torch with ONE thread (the
host-side deployment, stated), `gc` off inside the timed regions, `perf_counter`. Equality check before any timing: B's wiring choices give the same curves / wired counts as the stored card of the head (`ends_card_e358w_snap_nogeom_attach_thr0.1.json`), A the stored PRIMARY card. The host is checked before and after (1-minute load
average, other `run.py` jobs): QUIET when both loads are below 3 and no other job of the problem ran, else NOT QUIET (the numbers are then context only).
usage (problem folder):  wsl bash tools/env/_py14.sh ../temp/e358/time_wiring.py --seeds 0 1 [--rounds 3] [--max 0] [--out temp/e358/timing_wiring.json]
"""
import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jwire as JW  # noqa: E402
import jwire_head as JH  # noqa: E402
import qends_eval as QE  # noqa: E402
import qgraph_eval as QG  # noqa: E402
import qgraph_readout as QR  # noqa: E402


def host():
    la = open("/proc/loadavg").read().split()[0]
    ps = subprocess.run(["bash", "-c", "ps -eo cmd | grep -E 'run.py --id|jends_|jwire_|jdiag|jmaps_|qgraph' | grep -v grep | grep -v time_wiring | cut -c1-90"], capture_output=True, text=True).stdout.strip().splitlines()
    return dict(load1=float(la), other_jobs=ps[:8], cpus=os.cpu_count())


def prep(seed, nmax):
    dump = pickle.load(open(f"{ROOT}/temp/e276/predfull_e273_p_s{seed}.pkl", "rb"))
    ends = {f["row"]: f for f in pickle.load(open(f"{ROOT}/runs/e324_s_s{seed}/ends_val.pkl", "rb"))}
    spec = QR.parse("close:1.5@0.5")
    frames = []
    for d in dump[: nmax or None]:
        pred = QR.apply(dict(nodes=dict(pos=d["pos"], dirv=d["dirv"], e=d["e"], flags=d["flags"], section=d["section"], s=d.get("s")),
                             links=dict(src=d["link_src"], dst=d["link_dst"], p=d["link_p"], long=d["link_long"], a=d["link_a"], b=d["link_b"])), spec)
        graph, bk = QG.readout(pred, 0.5, 0.5)
        e = ends[d["row"]]
        k = e["kp_pos"].shape[0] // 2
        p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
        ia, ib = np.nonzero(p >= 0.1)
        pairs = QE.make_pairs(e, ia, ib, k, attach=True, chord=False)
        frames.append(dict(graph=graph, bk=bk, long=np.asarray(pred["links"]["long"], bool), pairs=pairs, kp_p=[(float(e["kp_p"][i]), float(e["kp_p"][k + j])) for i, j in zip(ia, ib)],
                           node_e=np.asarray(pred["nodes"]["e"], float).reshape(-1), link_p=np.asarray(pred["links"]["p"], float).reshape(-1)))
    return frames


def opts():
    return dict(snap=True, snap_nogeom=True)


def run_a(f):
    return QE.add_connectors(f["graph"].lines, f["graph"].edges, f["graph"].nodes, f["bk"]["kind"], f["bk"]["piece_links"], f["long"], f["pairs"], wire_choice=None, **opts())


def run_b(f, head, parts=None):
    t0 = time.perf_counter()
    keep_, ends_ = JW.kept_ends(f["graph"].lines, f["bk"]["kind"], f["bk"]["piece_links"], f["long"], f["pairs"], QE.cover_pos())
    attrs_ = JH.piece_attrs(f["graph"].lines, f["bk"]["kind"], f["bk"]["piece_nodes"], f["bk"]["piece_links"], f["node_e"], f["link_p"], keep_)
    t1 = time.perf_counter()
    wc = JH.wire_choice_head(head, ends_, attrs_, f["pairs"], f["kp_p"], QE.WIRE_POS)
    t2 = time.perf_counter()
    out = QE.add_connectors(f["graph"].lines, f["graph"].edges, f["graph"].nodes, f["bk"]["kind"], f["bk"]["piece_links"], f["long"], f["pairs"], wire_choice=wc, **opts())
    if parts is not None:
        parts.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
    return out


def timed(fn, *a):
    gc.disable()
    t = time.perf_counter()
    r = fn(*a)
    dt = time.perf_counter() - t
    gc.enable()
    return dt * 1000.0, r


def boot_ci(x, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default="temp/e358/timing_wiring.json")
    a = ap.parse_args()
    torch.set_num_threads(1)
    h0 = host()
    res = dict(label=None, protocol="per frame A (rule) vs B (head: candidate lists + head forward + add_connectors), median over rounds, paired mean difference with a bootstrap over frames; torch threads 1; host side only", host_before=h0, seeds={})
    for seed in a.seeds:
        frames = prep(seed, a.max)
        head = JH.load_wire(f"e358_w_s{seed}")
        # equality with the stored cards (curves, wired both ends)
        pa = json.load(open(f"{ROOT}/runs/e324_s_s{seed}/ends_card_snap_nogeom_attach_thr0.1.json"))
        pb = json.load(open(f"{ROOT}/runs/e324_s_s{seed}/ends_card_e358w_snap_nogeom_attach_thr0.1.json"))
        na, nb = (sum(r[1] for r in map(run_a, frames)), sum(r[2] for r in map(lambda f: run_b(f, head), frames))) if False else (None, None)
        ca = [run_a(f) for f in frames]
        cb = [run_b(f, head) for f in frames]
        eq = dict(curves_a=sum(r[1] for r in ca), wired_a=sum(r[2] for r in ca), curves_b=sum(r[1] for r in cb), wired_b=sum(r[2] for r in cb), stored_curves=pa["curves"], stored_wired_rule=pa["wired_both_ends"], stored_wired_head=pb["wired_both_ends"])
        eq["complete"] = (a.max == 0)
        if a.max == 0:
            assert eq["curves_a"] == eq["curves_b"] == eq["stored_curves"] and eq["wired_a"] == eq["stored_wired_rule"] and eq["wired_b"] == eq["stored_wired_head"], eq
        ta = np.zeros((a.rounds, len(frames)))
        tb = np.zeros((a.rounds, len(frames)))
        parts = np.zeros((a.rounds, len(frames), 3))
        for r in range(a.rounds):
            for i, f in enumerate(frames):
                first_a = ((r + i) % 2 == 0)                                                               # the order alternates by round and by frame parity
                pl = []
                if first_a:
                    ta[r, i] = timed(run_a, f)[0]
                    tb[r, i] = timed(lambda ff: run_b(ff, head, pl), f)[0]
                else:
                    tb[r, i] = timed(lambda ff: run_b(ff, head, pl), f)[0]
                    ta[r, i] = timed(run_a, f)[0]
                parts[r, i] = np.array(pl[0]) * 1000.0
        ma, mb = np.median(ta, 0), np.median(tb, 0)
        d = mb - ma
        mean, lo, hi = boot_ci(d)
        pt = np.median(parts, 0)
        res["seeds"][str(seed)] = dict(frames=len(frames), rounds=a.rounds, A_rule_ms=dict(mean=float(ma.mean()), median=float(np.median(ma)), p90=float(np.percentile(ma, 90))),
                                       B_head_ms=dict(mean=float(mb.mean()), median=float(np.median(mb)), p90=float(np.percentile(mb, 90))), B_minus_A_ms=dict(mean=mean, ci95=[lo, hi]),
                                       B_parts_ms_mean=dict(candidate_lists_and_attrs=float(pt[:, 0].mean()), head_forward=float(pt[:, 1].mean()), add_connectors=float(pt[:, 2].mean())), equality=eq,
                                       pairs_per_frame=float(np.mean([len(f["pairs"]) for f in frames])))
        s = res["seeds"][str(seed)]
        print(f"seed {seed}: {len(frames)} frames x {a.rounds} rounds; A rule {s['A_rule_ms']['mean']:.3f} ms/frame (median {s['A_rule_ms']['median']:.3f}), B head {s['B_head_ms']['mean']:.3f} ms/frame (median {s['B_head_ms']['median']:.3f}); B - A {mean:+.3f} ms [{lo:+.3f}, {hi:+.3f}]; "
              f"B parts: lists+attrs {s['B_parts_ms_mean']['candidate_lists_and_attrs']:.3f}, head {s['B_parts_ms_mean']['head_forward']:.3f}, add_connectors {s['B_parts_ms_mean']['add_connectors']:.3f}; {s['pairs_per_frame']:.1f} pairs / frame; equality {eq}", flush=True)
    h1 = host()
    res["host_after"] = h1
    res["label"] = "QUIET" if (h0["load1"] < 3 and h1["load1"] < 3 and not h0["other_jobs"] and not h1["other_jobs"]) else "NOT QUIET"
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print("HOST", res["label"], "load before / after", h0["load1"], h1["load1"], "other jobs before:", h0["other_jobs"][:3], "after:", h1["other_jobs"][:3])


if __name__ == "__main__":
    main()
