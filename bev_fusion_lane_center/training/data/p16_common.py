"""Loop 16 harness: shared constants, graph conversion and the metric card (one definition for every run, control and designs).
Graph format = graph_eval format: dict(pts=[(n,2) polylines in travel direction], edges=[(i, j)], nodes=[vertex ids per polyline]).
Usage: imported by p16_ctrl.py / p16_train.py / p16_diag.py / p16_triage.py."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_eval as GE  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
LADDER = f"{OUT}/e123_ladder"
HOME = os.path.expanduser("~")
D_FEAT = f"{HOME}/exp_gcar_b123"
N_VAL, N_TEST, N_TRAIN = 873, 833, 8414
from sparse_metrics import DP_ORACLE, KEEP, card, composite, frontier_scalars, graph_from_vertices, vert100  # noqa: E402,F401  (moved to evaluation/sparse_lanes)


def val_subset(n=250, total=N_VAL):
    """fixed VAL subset of the triage rungs 0-1 (same frames for every design); the full VAL (873) is used from rung 2"""
    return [int(i * total / n) for i in range(n)]


# ------------------------------------------------------------------ metrics


# ------------------------------------------------------------------ io
def write_card(run_id, payload, root=LADDER):
    os.makedirs(f"{root}/cards", exist_ok=True)
    p = f"{root}/cards/{run_id}.json"
    json.dump(payload, open(p, "w"), indent=1, default=float)
    return p


