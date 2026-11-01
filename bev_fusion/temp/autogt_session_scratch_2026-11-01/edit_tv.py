import re
# ---- suite.py: light option
p = "autogt/suite.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    options: stationary_fix, map_offsets, use_flow, use_radar, flow_config (autogt.flow.Config)\n",
              "    options: stationary_fix, map_offsets, use_flow, use_radar, flow_config (autogt.flow.Config)\n"
              "    light=True: a memory-light devkit (CAM_FRONT + radars only, autogt/light_nusc.py): v1.0-trainval in ~4 GB instead of 8+ GB\n")
s = s.replace("    def __init__(self, dataroot, version, grid=None, ss=4, **options):\n        super().__init__(os.path.expanduser(dataroot), version, **options)",
              "    def __init__(self, dataroot, version, grid=None, ss=4, light=False, **options):\n"
              "        dataroot = os.path.expanduser(dataroot)\n"
              "        if light:\n"
              "            from .light_nusc import light_nuscenes\n"
              "            options[\"nusc\"] = light_nuscenes(dataroot, version)\n"
              "        super().__init__(dataroot, version, **options)")
open(p, "w", encoding="utf-8").write(s)

# ---- checks.py: Checks(grid), feed(..., sample_time_us), merge
p = "autogt/checks.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    """accumulator of the checks above; feed(scene_name, k, token, gt_dict) per keyframe, then text() / scenes()."""

    def __init__(self, gen):
        self.gen, self.st = gen, collections.defaultdict(list)''', '''    """accumulator of the checks above; feed(scene_name, k, token, gt_dict, annotation_time_us) per keyframe, then text() /
    scene_rows(); small and picklable (workers return it, `merge` adds another one)."""

    def __init__(self, grid):
        self.grid, self.st = grid, collections.defaultdict(list)''')
s = s.replace("    def feed(self, name, k, tok, gk):\n        g, st = self.gen.grid, self.st", "    def feed(self, name, k, tok, gk, sample_time_us):\n        g, st = self.grid, self.st")
s = s.replace('''        st["dt_ms"].append((m["timestamp"] - self.gen.nusc.get("sample", tok)["timestamp"]) * 1e-3)''', '''        st["dt_ms"].append((m["timestamp"] - sample_time_us) * 1e-3)''')
s = s.replace('''    def text(self, fov=None):''', '''    def merge(self, other):
        """add the accumulated results of another Checks (e.g. of a worker that did other sequences)."""
        for k, v in other.st.items():
            self.st[k].extend(v)
        self.flags.update(other.flags)
        for n, c in other.per_scene.items():
            self.per_scene[n] = c
        return self

    def text(self, fov=None):''')
s = s.replace('''    ck = Checks(gen)
    for name in (scenes or gen.scenes()):
        for k, tok in enumerate(gen.keyframes(name)):
            if k % stride == 0:
                ck.feed(name, k, tok, gen.keyframe(tok))''', '''    ck = Checks(gen.grid)
    for name in (scenes or gen.scenes()):
        for k, tok in enumerate(gen.keyframes(name)):
            if k % stride == 0:
                ck.feed(name, k, tok, gen.keyframe(tok), gen.nusc.get("sample", tok)["timestamp"])''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
