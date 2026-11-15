"""scratch: how much do the outputs move when rad_mix wakes (B4 of test_radar_recursive failed at the threshold 1e-4)"""
import os, runpy, sys
sys.argv = ["x"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "test_radar_recursive.py")).read()
head = src[:src.index('print("A - D:')]
ns = {"__name__": "probe", "__file__": os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "test_radar_recursive.py"))}
exec(compile(head, ns["__file__"], "exec"), ns)
import torch
mem, ref = ns["twins"]()
k1r = ns["run"](ref, ns["mk"]([0, 1, 2]))
with torch.no_grad():
    e2 = mem.radar(ns["RAD"][2], ns["B"])
    for std in (0.05, 0.5):
        torch.manual_seed(1)
        mem.rad_mix.weight.normal_(0, std)
        a = ns["run"](mem, ns["mk"]([0, 1, 2]))
        print("std", std, "outputs gap", ns["gap"](a, k1r, state=False), "rad_fused gap", float((a["state"]["buf"]["rad_fused"] - e2).abs().max()))
