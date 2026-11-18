"""Checkpoint + VAL frames -> delivered lane graphs as JSON (the TEST split is not offered: it needs a logged grant, hard rule 1).

    bash tools/env/_py14.sh training/lanegraph/scripts/lg_infer.py --seed 0 --first 20 --out graphs.json [--device cuda] [--attached-ends]
"""
import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))     # the lanegraph modules (imported flat, the project's convention)
import torch  # noqa: E402

from lg_config import CHAMPION  # noqa: E402
from lg_features import FeatureStore, split_rows  # noqa: E402
from lg_graph import write_json  # noqa: E402
from lg_model import LaneGraphNet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--first", type=int, default=20, help="the first N VAL frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--attached-ends", action="store_true", help="connectors between the learned attached ends (still wired by the proximity rule)")
    a = ap.parse_args()
    cfg = dataclasses.replace(CHAMPION, attached_ends=a.attached_ends)
    net = LaneGraphNet.from_runs(a.seed, cfg).to(a.device)
    store, rows = FeatureStore(), split_rows("val")[: a.first]
    graphs, t0 = {}, time.perf_counter()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        for r, g in zip(chunk, net.graphs(store(chunk).to(a.device))):
            graphs[store.tokens[r]] = g
    write_json(a.out, graphs, cfg.dp_tol)
    print(f"{len(graphs)} frames -> {a.out} ({1000 * (time.perf_counter() - t0) / max(len(graphs), 1):.0f} ms per frame incl. loading and read-out)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
