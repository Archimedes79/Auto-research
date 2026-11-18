"""U3-U5 scratch: the three S4 gate command lines of temp/e372/gate372b.sh through `jlive_train.parse_args` and `resolve` (the head run's args checked against the removed options, the real runs/; reads only)"""
import os
import sys

import jlive_train as T

ROOT = os.environ["BFLC_ROOT"]
for arm, x in (("z", ["--train-nodes", "0"]), ("b", ["--train-nodes", "1", "--live-inputs", "0"]), ("s4", ["--train-nodes", "1", "--live-inputs", "1"])):
    a = T.parse_args(["--name", f"e372_gate_{arm}_s0", "--seed", "0", "--head-run", "e373_cb_s0"] + x + ["--steps", "50", "--bs", "8", "--lr", "1e-4", "--workers", "4"])
    args = T.resolve(a, os.path.join(ROOT, "runs"))
    print(f"gate {arm}: parsed and resolved: train_nodes {args['train_nodes']} live_inputs {args['live_inputs']} none_weight {args['none_weight']} pos_loss {args['pos_loss']} head_seed {args['head_seed']} "
          f"b_run {args['b_run']}; T from {T.__file__}", flush=True)
print("gate parse ok", file=sys.stderr)
