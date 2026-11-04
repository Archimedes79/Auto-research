"""Paired scene bootstrap of the maps-only run against its controls (card: experiments/t6_soa_gap/experiment.md). CPU.
a -> b = b - a: ref_s0 -> maps_only, ref_s1 -> maps_only, ref_s0 -> ref_s1 (seed spread), ref_s0 -> ref_long (value of doubling the steps).
Usage (bev_fusion folder): python temp/soa_gap/compare_maps_only.py"""
import sys

sys.path.insert(0, ".")
from evaluation.bev_legacy import compare  # noqa: E402

for a, b in (("runs/t2_ref_s0", "runs/t6_maps_only"), ("runs/t2_ref_s1", "runs/t6_maps_only"),
             ("runs/t2_ref_s0", "runs/t2_ref_s1"), ("runs/t2_ref_s0", "runs/t3_ref_long")):
    compare(a, b)
    print()
