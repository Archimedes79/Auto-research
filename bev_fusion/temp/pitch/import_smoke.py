"""Import smoke test after the move of autogt out of bev_fusion: the imports of the like-for-like map scoring and of the evaluation, with the same sys.path as the scripts use.
Run from bev_fusion (WSL, gcar python, CPU): python temp/pitch/import_smoke.py"""
import os
import sys

sys.path.insert(0, "training")
sys.path.insert(0, "evaluation")
sys.path.insert(0, ".")
import torch  # noqa: E402

torch.set_num_threads(2)
from evaluation.bev_probe import add_sample, layer_index, load_run, masks, predictions  # noqa: E402,F401
from metrics import SegMeter  # noqa: E402,F401
from train import forward_loss, make_datasets, make_loader  # noqa: E402,F401
import nuscenes_dataset  # noqa: E402
import autogt  # noqa: E402

print("autogt from", os.path.dirname(os.path.abspath(autogt.__file__)))
print("nuscenes_dataset from", os.path.abspath(nuscenes_dataset.__file__))
print("imports ok")
