"""E385 no-run check (card: "the delivered recipe is shown to reproduce on the simplified code by a CPU 20-step equality of its command line against the archive tag's code"):
stage 2 of the D120 champion (jlive_train s4 on the head run e373_cb_s<S>) for 20 steps on the CPU, in a throw-away runs folder; prints the sha256 of the final
head state, of B''s node-head state and the logged losses. Run twice through ux_run_with.py: once with U2_WT = the live tree (the landed code), once with U2_WT =
an extract of tag e3df44fb (the code before the simplification); the two printed lines must be equal.
usage: _py14.sh temp/simplify/ux_run_with.py temp/e385/eq_s4.py <seed>"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

import torch

import jlive_train as L
import jtoken_train as JKT

S = int(sys.argv[1])
torch.set_num_threads(2)
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # the runner's cwd is the script's folder


def sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp", "e385")) as tmp:
    os.makedirs(os.path.join(tmp, f"e373_cb_s{S}"))
    shutil.copy(os.path.join(ROOT, "runs", f"e373_cb_s{S}", "model.pt"), os.path.join(tmp, f"e373_cb_s{S}", "model.pt"))
    cli = ["--name", "eq", "--seed", str(S), "--head-run", f"e373_cb_s{S}", "--b-run", os.path.join(ROOT, "runs", f"e292_a_s{S}"),
           "--train-nodes", "1", "--live-inputs", "1", "--steps", "20", "--bs", "2", "--lr", "1e-4", "--workers", "0"]
    L.train(L.parse_args(cli), runs=tmp, ckpt_every=500)
    ck = torch.load(os.path.join(tmp, "eq", "model.pt"), map_location="cpu", weights_only=False)
    rows = [json.loads(l) for l in open(os.path.join(tmp, "eq", "log.jsonl"))]
    losses = [round(r["total"], 6) for r in rows]
    print(f"EQ seed {S}: code {L.__file__} | head {sha(ck['state'])} | b_state {sha(ck['b_state'])} | logged total {losses}", flush=True)
