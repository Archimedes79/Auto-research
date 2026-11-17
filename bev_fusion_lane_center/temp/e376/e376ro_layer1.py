"""E376 read-out, layer 1 (throw-away): the promoted anatomy on the stored dump runs/e373_cb_s0/token_val.pkl must give the numbers of the E376 card (12,509 delivered long links, 7,748 GT long links, shares tp 0.171 / dup 0.130 /
wrong 0.339 / false 0.360); the json goes to temp/e376/, never into the run folder. Argument: the module to import (default e376ro_jtoken_eval, the scratch copy; jtoken_eval = the promoted one)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
E = __import__(sys.argv[1] if len(sys.argv) > 1 else "e376ro_jtoken_eval")
out = os.path.join(HERE, sys.argv[2]) if len(sys.argv) > 2 else HERE
os.makedirs(out, exist_ok=True)
path = E.anatomy(E.parse_args(["anatomy", "--run", "e373_cb_s0", "--seed", "0"]), out_dir=out)
d = json.load(open(path))
want = dict(delivered=12509, gt_long_links=7748, tp=0.171, dup_of_true=0.130, wrong_pair=0.339, false_end=0.360)
got = dict(delivered=d["delivered"], gt_long_links=d["gt_long_links"], **{k: round(d["categories"][k]["share"], 3) for k in E.LONG_CATS})
print(json.dumps(dict(want=want, got=got, exact_shares={k: d["categories"][k]["share"] for k in E.LONG_CATS}), indent=1))
print(("PASS" if got == want else "FAIL") + f": layer 1 of the anatomy against the E376 card ({os.path.relpath(path, os.environ['BFLC_ROOT'])})")
