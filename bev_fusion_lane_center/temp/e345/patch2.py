R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/"
p = R + "jprune_probe.py"
s = open(p, encoding="utf-8").read()
old = '''        self.run, self.dev, self.out, self.log = run, dev, out_dir, log
'''
new = '''        self.run, self.dev, self.out, self.log = run, dev, out_dir, log
        wait_for_gpu("jprune_probe sweep", dev)
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    model, _ = JEV.load_model(run, dev)
    cache = build_cache("val", workers=0, max_frames=8 * n_batches)'''
new = '''    wait_for_gpu("jprune_probe time", dev)
    model, _ = JEV.load_model(run, dev)
    cache = build_cache("val", workers=0, max_frames=8 * n_batches)'''
assert old in s
s = s.replace(old, new, 1)
old = '''def time_pass(run, dev, n_batches=4):'''
new = '''def wait_for_gpu(who, dev):
    """D100: a GPU step does not start next to two trainings (`gpu_gate.wait_for_room`: waits while the GPU memory in use is above its limit, never dead-locks); no effect on the CPU"""
    if str(dev).startswith("cuda"):
        import gpu_gate
        gpu_gate.wait_for_room(who=who)


def time_pass(run, dev, n_batches=4):'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
