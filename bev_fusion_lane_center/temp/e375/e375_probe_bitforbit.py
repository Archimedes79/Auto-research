"""E375 scratch probe (one use, deleted after): the HEAD module (e375_jtoken_eval_old.py) and the new one (e375_jtoken_eval_new.py) on the same 3-frame dump of the untrained head: score (record, variant), sweep (default grid) and graphs
(record, variant) without the new options -> every file equal (cards without date / commit)"""
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


OLD, NEW = load("jte_old", "e375_jtoken_eval_old.py"), load("jte_new", "e375_jtoken_eval_new.py")
import torch  # noqa: E402

torch.set_num_threads(2)
SMOKE = os.path.join(NEW.ROOT, "runs", "e365_smoke_build", "model.pt")
with tempfile.TemporaryDirectory() as tmp:
    runs = os.path.join(tmp, "runs")
    os.makedirs(os.path.join(runs, "h"))
    ck = torch.load(SMOKE, map_location="cpu", weights_only=False)
    ck["state"] = NEW.floor_model().state_dict()
    torch.save(ck, os.path.join(runs, "h", "model.pt"))
    NEW.dump(NEW.parse_args(["dump", "--run", "h", "--seed", "0", "--cpu", "--max", "3"]), runs)
    outs = {}
    for tag, E in (("old", OLD), ("new", NEW)):
        scratch = os.path.join(tmp, "w")                                  # the same folder for both (the cards name the record card's path); emptied between them
        shutil.rmtree(scratch, ignore_errors=True)
        out = os.path.join(scratch, "out")
        os.makedirs(out)
        shutil.copy(os.path.join(NEW.ROOT, "temp", "e365", "planb_s0.pkl"), os.path.join(scratch, "planb_s0.pkl"))
        E.SCRATCH = scratch
        E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3"]), runs, out)
        E.score(E.parse_args(["score", "--run", "h", "--seed", "0", "--max", "3", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs, out)
        E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3"]), runs, out)
        g = os.path.join(scratch, "g", "graphs.pkl")
        os.makedirs(os.path.dirname(g))
        E.graphs(E.parse_args(["graphs", "--run", "h", "--seed", "0", "--max", "3", "--out", g, "--name", "S1"]), runs)
        E.graphs(E.parse_args(["graphs", "--run", "h", "--seed", "0", "--max", "3", "--out", g, "--name", "S1_v", "--link-thr", "0.1", "--keep-thr", "0.6", "--no-none-gate"]), runs)
        files = {f: json.load(open(os.path.join(out, f))) for f in sorted(os.listdir(out))}
        for f, c in files.items():
            c.pop("date", None), c.pop("commit", None)
        files["graphs.pkl"] = pickle.dumps(pickle.load(open(g, "rb")))
        files["perframe_table.json"] = open(os.path.join(scratch, "g", "perframe_table.json")).read()
        outs[tag] = files
    assert list(outs["old"]) == list(outs["new"]), (list(outs["old"]), list(outs["new"]))
    diff = [f for f in outs["old"] if outs["old"][f] != outs["new"][f]]
    print("files:", list(outs["old"]))
    print("BITFORBIT", "PASS" if not diff else f"FAIL {diff}")
