"""Read-out and scoring of the junction pair head v2 (E287), on the delivered graph of the lane head B (+ the E275 hard close-node step), the same way as the E284 end head (`qends_eval.py`).

    wsl bash tools/env/_py14.sh jends_eval.py dump  --run e287_p2_s0                      # GPU or CPU: keypoints and pair logits on the 873 VAL frames -> runs/<run>/ends_val.pkl, ends_head.json (recall / precision of ends and pairs)
    wsl bash tools/env/_py14.sh jends_eval.py score --run e287_p2_s0 --seed 0 [--thr 0.5]   # CPU: the delivered graph of B with the head's connectors, scored by the E273 scorer (`qends_eval.cmd_score`), D79's rule against the plain graph
    wsl bash tools/env/_py14.sh jends_eval.py selftest

The keypoints come from the junction maps of E286 (the pixel stage of the maps run is B's own pyramid, so the delivered graph under the connectors is B's: `score` is told not to compare tensors with B (the maps run
is a different checkpoint) but reads B's full prediction dump of the same seed). The rule is D79's, written in the card of E287 before any run; the verdict over the seeds is the card's.
"""
import argparse
import json
import os
import pickle
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
    m = JT.JEndModel(a["maps_args"], order=not a["no_order"], evidence=not a["no_evidence"], scales=bool(a.get("scales", False)))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), a


def dump(a):
    import lane_gt_data as LD
    import qendnet as QEN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, args = load_model(a.run, dev)
    LD.QData.DEFAULT_JUNCTION = True
    ds, dl = LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    frames, tot = [], {}
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            _, _, out = m(x, gt=None)
            for k, v in QEN.end_counts(out, b, a.thr).items():
                tot[k] = tot.get(k, 0) + v
            if a.max and len(frames) >= a.max:
                break
            for i in range(len(b["token"])):
                frames.append(dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=out["kp_pos"][i].cpu().numpy(), kp_dir=out["kp_dir"][i].cpu().numpy(), kp_p=out["kp_p"][i].cpu().numpy(),
                                   valid=out["valid"][i].cpu().numpy(), logit=out["logit"][i].float().cpu().numpy(), allowed=out["allowed"][i].cpu().numpy(),
                                   scale=out["scale"][i].float().cpu().numpy() if "scale" in out else None))
    r = lambda x, y: x / max(y, 1)  # noqa: E731
    head = dict(ent_recall=r(tot["ent_hit"], tot["ent_n"]), ent_precision=r(tot["ent_ok"], tot["ent_kp"]), ext_recall=r(tot["ext_hit"], tot["ext_n"]), ext_precision=r(tot["ext_ok"], tot["ext_kp"]),
                pair_recall=r(tot["pairs_hit"], tot["pairs_n"]), pair_precision=r(tot["pairs_ok"], tot["pairs_kp"]), counts=tot, thr=a.thr, frames=len(frames), order=not args["no_order"],
                evidence=not args["no_evidence"], maps=args["maps"])
    run_dir = os.path.join(ROOT, "runs", a.run)
    pickle.dump(frames, open(os.path.join(run_dir, "ends_val.pkl"), "wb"))
    json.dump(head, open(os.path.join(run_dir, "ends_head.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in head.items() if k != "counts"}, indent=1))


def score(a):
    import qends_eval as QE
    QE.cmd_score(SimpleNamespace(run=a.run, seed=a.seed, thr=a.thr, full=None, base=None, skip_base_check=True, chord=bool(getattr(a, "chord", False))))


def selftest():
    import jends_train as JT
    m = JT.JEndModel(dict(d=64, ctx="none"), order=True, evidence=True)
    x = torch.randn(1, 39, 200, 200)
    P, out, oe = m(x)
    assert oe["logit"].shape == (1, 32, 32) and out["j1"].shape == (1, 8, 100, 100)
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
        if name == "score":
            s.add_argument("--seed", type=int, required=True)
            s.add_argument("--chord", action="store_true", help="E303: the chord-tangent connectors even when the dump has learned handle scales (the shape control: same pairs)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"dump": dump, "score": score, "selftest": lambda _: selftest()}[a.cmd](a)
