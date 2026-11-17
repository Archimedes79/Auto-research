import os, sys, json, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(root, "training/train"))
import jtoken_train as JKT
with tempfile.TemporaryDirectory() as runs:
    JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
    rows = [json.loads(l) for l in open(os.path.join(runs, "r", "log.jsonl"))]
    print(rows[0].keys())
    print("LOSSES", [round(r["total"], 8) for r in rows if "total" in r])
