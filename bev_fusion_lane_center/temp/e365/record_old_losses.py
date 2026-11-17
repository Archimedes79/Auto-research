import os, sys, json, tempfile
os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(root, "training/train"))
import torch; torch.set_num_threads(2)
sys.path.insert(0, os.path.join(root, "temp", "e365"))
import jtoken_train_old as JKT
with tempfile.TemporaryDirectory(dir=os.path.join(root, "temp", "e365")) as runs:
    JKT.train(JKT.parse_args(["--name", "r", "--seed", "0", "--smoke", "--bs", "2", "--workers", "0"]), runs, 500, 48)
    rows = [json.loads(l) for l in open(os.path.join(runs, "r", "log.jsonl"))]
    print(rows[0].keys())
    print("LOSSES", [round(r["total"], 8) for r in rows if "total" in r])
    import torch
    sd = torch.load(os.path.join(runs, "r", "model.pt"), map_location="cpu", weights_only=False)["state"]
    print("ROW", {k: rows[-1][k] for k in ("total", "nll_s", "nll_l", "ce1", "ce3", "scale_l1")})
    print("SUM", repr(sum(float(v.double().abs().sum()) for k, v in sd.items() if v.dtype.is_floating_point)), len(sd))
