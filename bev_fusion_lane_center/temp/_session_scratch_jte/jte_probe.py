"""throw-away probe: are the --ref-cell mutants (keep / gate / swap) and the sweep's junction-point count visible on the test frames?"""
import os
import pickle
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "tests"))
import test_jtoken_eval as T  # noqa: E402

E, np, torch = T.E, T.np, T.torch
import qgraph_eval as QE  # noqa: E402

ds = E.open_cache("val", 0)
fr = QE.val_frames()
quiet = lambda *a, **k: None  # noqa: E731
with tempfile.TemporaryDirectory() as tmp:
    runs, scratch = os.path.join(tmp, "runs"), os.path.join(tmp, "scratch")
    os.makedirs(os.path.join(runs, "h"))
    os.makedirs(os.path.join(scratch, "out"))
    ck = torch.load(T.SMOKE, map_location="cpu", weights_only=False)
    ck["state"] = E.floor_model().state_dict()
    torch.save(ck, os.path.join(runs, "h", "model.pt"))
    E.dump(E.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--every", "3", "--max", "3"]), runs)
    d = pickle.load(open(os.path.join(runs, "h", "token_val_every3_n3.pkl"), "rb"))
    toks = [ds[i]["tokens_np"] for i in (0, 3, 6)]
    for f, t in zip(d["frames"], toks):
        pn = f["pnone"][t["valid"]]
        print("pnone: valid", len(pn), "in [0.5,0.6)", int(((pn >= 0.5) & (pn < 0.6)).sum()), "in [0.6,0.7)", int(((pn >= 0.6) & (pn < 0.7)).sum()), "< 0.5", int((pn < 0.5).sum()))

    def direct(lt, kt, gate, ltl, kt_w=None):
        res = E.score_preds([E.frame_prediction(t, E.decode_variant(f, t["valid"], lt, kt, gate, ltl), "rec", kt if kt_w is None else kt_w) for f, t in zip(d["frames"], toks)], [fr[i] for i in (0, 3, 6)], log=quiet)
        return res

    keys = E.KEYS + ("junction_P", "junction_R")
    for name, cell in (("ref", (0.1, 0.5, True, 0.2)), ("keep mutant", (0.1, 0.6, True, 0.2)), ("keep mutant decode only", (0.1, 0.6, True, 0.2, 0.5)), ("gate mutant", (0.1, 0.5, False, 0.2)),
                       ("swap", (0.2, 0.5, True, 0.1)), ("variant", (0.3, 0.6, False, 0.05))):
        r = direct(*cell)
        print(name, cell, {k: round(r["graph"][k], 6) for k in keys}, "junc pred/gt/matched", np.sum([c["lane"]["junc"] for c in r["rec"]["graph"]], 0).tolist(), "link_long P/R", r["nodes"].get("link_long_P"), r["nodes"].get("link_long_R"))
