import inspect, os, sys, textwrap
ROOT = "/mnt/c/Auto-research/bev_fusion"
src = open(ROOT + "/tests/test_stream_temporal.py").read()
head = src.split("# ------------------------------------------------------------------------------------------------ A - D: baseline")[0]
g = {"__file__": ROOT + "/tests/test_stream_temporal.py", "__name__": "tt"}
exec(compile(head, "head", "exec"), g)
m = g["build"](cam_buffer="recurrent")
b = g["mk"]([0, 1, 2])
vis = g["Precompute"].from_model(m)(dict(b), ("vis_cam",))["vis_cam"]
print("vis_cam shape", tuple(vis.shape), "mean", float(vis.mean()), "min", float(vis.min()))
