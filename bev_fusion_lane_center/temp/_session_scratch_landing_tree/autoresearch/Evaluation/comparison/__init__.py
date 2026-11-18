"""The comparison core of the evaluation suite: where a prediction is compared with the ground truth. See README.md in this folder.

    from Evaluation.comparison import maps, boxes          # the core: plain numpy arrays and dicts in, counts and numbers out
    from Evaluation.comparison.metrics import MapMetric, BoxMetric      # the same as ready-made Metrics of the suite
"""
from . import boxes, maps

__all__ = ["maps", "boxes"]
