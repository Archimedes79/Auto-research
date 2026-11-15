"""Read-out and scoring of the junction pair head v2 (E287), on the delivered graph of the lane head B (+ the E275 hard close-node step), the same way as the E284 end head (`qends_eval.py`).

    wsl bash tools/env/_py14.sh jends_eval.py dump  --run e287_p2_s0                      # GPU or CPU: keypoints and pair logits on the 873 VAL frames -> runs/<run>/ends_val.pkl, ends_head.json (recall / precision of ends and pairs)
    wsl bash tools/env/_py14.sh jends_eval.py score --run e287_p2_s0 --seed 0 [--thr 0.5]   # CPU: the delivered graph of B with the head's connectors, scored by the E273 scorer (`qends_eval.cmd_score`), D79's rule against the plain graph
    wsl bash tools/env/_py14.sh jends_eval.py selftest

The keypoints come from the junction maps of E286 (the pixel stage of the maps run is B's own pyramid, so the delivered graph under the connectors is B's: `score` is told not to compare tensors with B (the maps run
is a different checkpoint) but reads B's full prediction dump of the same seed). The rule is D79's, written in the card of E287 before any run; the verdict over the seeds is the card's.
"""
import argparse
import dataclasses
import json
import os
import pickle
import shutil
import sys
from types import SimpleNamespace

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))


def load_model(run, dev):
    import jends_train as JT
    ck = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")
    a = ck["args"]
    m = JT.JEndModel(a["maps_args"], order=not a["no_order"], evidence=not a["no_evidence"], scales=bool(a.get("scales", False)), ntok=bool(a.get("ntok", False)), unc=bool(a.get("unc", False)), attach=bool(a.get("attach", False)), prior=bool(a.get("lane_prior", False)))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), a


def dump(a):
    import lane_gt_data as LD
    import qendnet as QEN
    import jprior as JP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                                                                                                          # D100: a dump / score next to two trainings crashed one of them (CUDA unknown error, 2026-11-14)
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jends_eval dump {a.run}")
    m, args = load_model(a.run, dev)
    pmode, pnoise = getattr(a, "prior_mode", None), getattr(a, "prior_noise", 0.0)                                                              # (a caller with its own Namespace has neither)
    specs = JP.readouts(pmode or "clean", pnoise) if m.je.prior is not None else []                                              # E291: a model with the lane prior has one read-out per spec, all from ONE pass over the loader
    assert m.je.prior is not None or (pmode is None and not pnoise), f"{a.run} has no lane prior: --prior-mode / --prior-noise do not apply"
    run_dirs = [] if specs else [os.path.join(ROOT, "runs", a.run)]
    for sp in specs:                                                                                                                           # E291: a read-out is a NEW run folder (a copy of the model and its config, its own dump and cards), never an overwritten one
        d = os.path.join(ROOT, "runs", f"{a.run}_{sp.name}")
        os.makedirs(d, exist_ok=False)
        for f in ("model.pt", "config.json"):
            shutil.copy2(os.path.join(ROOT, "runs", a.run, f), os.path.join(d, f))
        run_dirs.append(d)
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PRIOR = tuple(specs)
    ds, dl = LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    R = max(1, len(specs))
    frames, tot, psum = [[] for _ in range(R)], [{} for _ in range(R)], [dict(term=0.0, tok=0.0, gate=0.0, n=0.0) for _ in range(R)]
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            outs = [m(x, gt=None, arms=({k: v.to(dev) for k, v in b["arms"][0].items()} if specs else None))[2] for r in range(R)]
            for r, out in enumerate(outs):
                for k, v in QEN.end_counts(out, b, a.thr).items():
                    tot[r][k] = tot[r].get(k, 0) + v
            if a.max and len(frames[0]) >= a.max:
                break
            for r, out in enumerate(outs):
                if specs:                                                                                                                      # E291: the size of the prior term and the gate over the valid keypoints
                    vm = out["valid"].flatten(1).float()
                    psum[r]["term"] += float((out["prior"]["term"] * vm).sum())
                    psum[r]["tok"] += float((out["prior"]["tok"] * vm).sum())
                    psum[r]["gate"] += float((out["prior"]["gate"] * vm).sum())
                    psum[r]["n"] += float(vm.sum())
                for i in range(len(b["token"])):
                    fr = dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=out["kp_pos"][i].cpu().numpy(), kp_dir=out["kp_dir"][i].cpu().numpy(), kp_p=out["kp_p"][i].cpu().numpy(),
                              valid=out["valid"][i].cpu().numpy(), logit=out["logit"][i].float().cpu().numpy(), allowed=out["allowed"][i].cpu().numpy(),
                              scale=out["scale"][i].float().cpu().numpy() if "scale" in out else None)
                    if "kp_logsig" in out:                                                                                                         # E318: the position scale (log sigma, m) of every keypoint
                        fr["logsig"] = out["kp_logsig"][i].float().cpu().numpy()
                    if "att_pos" in out:                                                                                                           # E320: the attached ends (position, heading, probability of an attachment) of every keypoint
                        fr.update(att_pos=out["att_pos"][i].float().cpu().numpy(), att_dir=out["att_dir"][i].float().cpu().numpy(), att_p=out["att_p"][i].float().cpu().numpy())
                    if specs:                                                                                                                      # E291: per keypoint the gate, the norm of the term and of the token, the count that was shown (-1 = unknown or absent) and its arm
                        fr["prior"] = {k: out["prior"][k][i].cpu().numpy() for k in ("gate", "term", "tok", "n", "arm")}
                    frames[r].append(fr)
    rr = lambda x, y: x / max(y, 1)  # noqa: E731
    for r, run_dir in enumerate(run_dirs):
        head = dict(ent_recall=rr(tot[r]["ent_hit"], tot[r]["ent_n"]), ent_precision=rr(tot[r]["ent_ok"], tot[r]["ent_kp"]), ext_recall=rr(tot[r]["ext_hit"], tot[r]["ext_n"]), ext_precision=rr(tot[r]["ext_ok"], tot[r]["ext_kp"]),
                    pair_recall=rr(tot[r]["pairs_hit"], tot[r]["pairs_n"]), pair_precision=rr(tot[r]["pairs_ok"], tot[r]["pairs_kp"]), counts=tot[r], thr=a.thr, frames=len(frames[r]), order=not args["no_order"],
                    evidence=not args["no_evidence"], maps=args["maps"])
        if specs:                                                                                                                              # E291: the read-out and the size of the prior term: sum of ||sigmoid(g) f|| over the sum of ||t|| of the valid keypoints (the kill criterion: not zero), the mean gate
            head["prior"] = dict(readout=specs[r].name, spec=dataclasses.asdict(specs[r]), term_over_token=psum[r]["term"] / max(psum[r]["tok"], 1e-9), gate_mean=psum[r]["gate"] / max(psum[r]["n"], 1.0), source_run=a.run)
        pickle.dump(frames[r], open(os.path.join(run_dir, "ends_val.pkl"), "wb"))
        json.dump(head, open(os.path.join(run_dir, "ends_head.json"), "w"), indent=1)
        print(run_dir, json.dumps({k: v for k, v in head.items() if k != "counts"}, indent=1))


def score(a):
    import qends_eval as QE
    ck = torch.load(os.path.join(ROOT, "runs", a.run, "model.pt"), map_location="cpu")
    args = ck["args"]
    mp = str(args.get("maps", ""))
    import re
    m = re.search(r"_s(\d+)[/\\]", mp.replace("\\", "/") + "/")
    assert args.get("seed", a.seed) == a.seed and (m is None or int(m.group(1)) == a.seed), f"--seed {a.seed} does not match the run (seed {args.get('seed')}, maps {mp}): its delivered graph would be another seed's"
    base_pt = getattr(a, "base", None) or os.path.join(ROOT, "runs", f"e273_p_s{a.seed}", "model.pt")                          # E292: the checkpoint of the head whose graph is delivered (B')
    base = torch.load(base_pt, map_location="cpu")["state"]
    keys = [k for k in base if k.startswith("pyr.") and (("jmodel." + k) in ck["state"] or k in ck["state"])]
    assert keys, "no pyramid tensor of the base run is in the checkpoint: the pyramid check would be empty"
    pyr_same = all(torch.equal(ck["state"]["jmodel." + k] if "jmodel." + k in ck["state"] else ck["state"][k], v) for k, v in base.items() if k.startswith("pyr.") and (("jmodel." + k) in ck["state"] or k in ck["state"]))
    print("pyramid of the maps run == the base run's:", pyr_same, "(False = a joint pyramid: the delivered graph is B's, the connectors come from other pixel features: label the result 'maps-only')")
    assert pyr_same or not getattr(a, "base", None), "E292: the pyramid of the maps run is not the pyramid of --base: the connectors would sit on other pixel features than the delivered graph"
    QE.cmd_score(SimpleNamespace(run=a.run, seed=a.seed, thr=a.thr, full=getattr(a, "full", None), base=getattr(a, "base", None), spec=getattr(a, "spec", None), tag=getattr(a, "tag", None), wire=getattr(a, "wire", None), wire_run=getattr(a, "wire_run", None), skip_base_check=True, chord=bool(getattr(a, "chord", False)), snap=bool(getattr(a, "snap", False)), snap_all=bool(getattr(a, "snap_all", False)), snap_guard=bool(getattr(a, "snap_guard", False)), snap_dedup=bool(getattr(a, "snap_dedup", False)), snap_nogeom=bool(getattr(a, "snap_nogeom", False)), snap_both=bool(getattr(a, "snap_both", False)), attach=bool(getattr(a, "attach", False)), attach_r=getattr(a, "attach_r", None)))


def selftest():
    import jends_train as JT
    m = JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True)
    x = torch.randn(1, 39, 200, 200)
    P, out, oe = m(x)
    assert oe["logit"].shape == (1, 32, 32) and out["j1"].shape == (1, 8, 100, 100)
    # E305: with e2e the pair logits reach the maps head (and never the frozen pyramid); without it the maps head gets nothing
    for e2e in (False, True):
        m = JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True, e2e=e2e)
        m.zero_grad()
        P, out, oe = m(torch.randn(1, 39, 200, 200), gt=dict(ent=torch.zeros(1, 16, 4), ext=torch.zeros(1, 16, 4), ent_m=torch.zeros(1, 16, dtype=torch.bool), ext_m=torch.zeros(1, 16, dtype=torch.bool)))
        (oe["logit"].sum() + oe["kp_pos"].sum()).backward()
        gm = sum(float(p.grad.abs().sum()) for p in m.jmodel.jm.parameters() if p.grad is not None)
        gp = sum(float(p.grad.abs().sum()) for p in m.jmodel.pyr.parameters() if p.grad is not None)
        assert gp == 0.0, "the frozen pyramid must not get a gradient"
        assert (gm > 0) == e2e, (e2e, gm)
    mp = JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True, prior=True)                                                           # E291: the prior head as the dump builds it: zero term at the start, the table in, the numbers of the prior out
    import jprior as JP
    import numpy as np
    arms = JP.collate([dict(c=np.zeros((1, 2), "float32"), t=np.array([[0.0, 1.0]], "float32"), n_in=np.array([2]), n_out=np.array([2]))])
    _, _, op = mp(torch.randn(1, 39, 200, 200), arms=arms)
    assert op["prior"]["term"].shape == (1, 64) and float(op["prior"]["term"].abs().max()) == 0.0 and op["prior"]["tok"].shape == (1, 64), "the prior term is zero at the start"
    print("jends_eval selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    for name in ("dump", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--thr", type=float, default=0.5)
        if name == "dump":
            s.add_argument("--max", type=int, default=0, help="only the first N frames (a test of the pipeline; the score needs all 873)")
            s.add_argument("--prior-mode", default=None, help="E291, a model trained with --lane-prior: the read-outs of the arm table, a comma list of clean (default), noise10, noise30, pm1, u, s or `noise` with --prior-noise r: ONE pass over the loader, one NEW run folder <run>_<name> per read-out (model.pt and config.json copied, ends_val.pkl, ends_head.json); u = every count unknown, geometry kept; s = the table of another frame")
            s.add_argument("--prior-noise", type=float, default=0.0, help="E291: the rate of wrong counts of --prior-mode noise (0.1 -> folder <run>_noise10)")
        if name == "score":
            s.add_argument("--seed", type=int, required=True)
            s.add_argument("--chord", action="store_true", help="E303: the chord-tangent connectors even when the dump has learned handle scales (the shape control: same pairs)")
            s.add_argument("--snap", action="store_true", help="E311: connectors drawn from the end of the delivered piece wired to the entry to the start of the piece wired to the exit (writes ends_card_snap_thr<t>.json, the card of record stays)")
            s.add_argument("--attach-r", type=float, default=None, help="E320: with --attach, the radius (m) of the proximity wiring of the connector ends (default 4: every piece end in reach; the tight read-outs use 1.5 and 1.0); the card file name carries it")
            s.add_argument("--full", default=None, help="E292: the full prediction dump of the head whose graph is delivered (B'); needs --base and --tag")
            s.add_argument("--base", default=None, help="E292: the checkpoint of that head; its pyramid must be the maps run's (asserted); needs --full")
            s.add_argument("--spec", default=None, help="E292: the read-out of the delivered graph (default close:1.5@0.5; plain@0.5 for a head with the Matrix-NMS layer); needs --tag")
            s.add_argument("--wire", default=None, choices=(None, "twin", "head"), help="E325: the zero-parameter twin of the nearest-piece rule (twin) or the learned head of --wire-run (head) chooses the wired pieces (needs --snap and --tag)")
            s.add_argument("--wire-run", default=None, help="E325: the run of the head for --wire head")
            s.add_argument("--tag", default=None, help="E292: names the card (ends_card_<tag>_...): mandatory with --full / --base / --spec; a tagged card is never overwritten")
            s.add_argument("--attach", action="store_true", help="E320: connectors drawn from the ATTACHED ends of the dump (att_pos / att_dir; no snap, the default wiring of every piece in reach), writes ends_card_attach_thr<t>.json")
            for k_ in ("--snap-all", "--snap-guard", "--snap-dedup", "--snap-nogeom", "--snap-both"):
                s.add_argument(k_, action="store_true", help="refinement of --snap (see qends_eval.add_connectors)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"dump": dump, "score": score, "selftest": lambda _: selftest()}[a.cmd](a)
