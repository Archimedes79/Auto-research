"""run only the CPU spec part of lane_grid/selftest.py (the rest needs the GPU)"""
import os, sys
src = open(os.path.join(os.environ["BFLC_ROOT"], "evaluation/lane_grid/selftest.py")).read()
head = src.split("spec_hysteresis_with_range()\nprob")[0] + "spec_hysteresis_with_range()\n"
exec(compile(head, "lane_grid_selftest_head", "exec"))
