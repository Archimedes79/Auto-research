p = "autogt/nuscenes_adapter.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import collections\nimport os\n", "import collections\nimport os\nfrom concurrent.futures import ThreadPoolExecutor\n", 1)
s = s.replace("        self._radar_cache = collections.OrderedDict()\n        self._max_cache = max_radar_cache\n",
              "        self._radar_cache = collections.OrderedDict()\n        self._max_cache = max_radar_cache\n        self._points = {}                                  # radar files read ahead by `_prefetch`, taken by `_radar_frame`\n", 1)
old = s[s.index("    def radar(self, t, only=None):"):s.index("    def _radar_frame(self, sd, ch, RadarPointCloud):")]
new = '''    def radar(self, t, only=None):
        """RadarFrames of the five radars (or of `only`, a list of channel names) within +-radar_window s of t."""
        from nuscenes.utils.data_classes import RadarPointCloud
        nusc = self.nusc
        sample = self._nearest_sample(t)
        want = []
        for ch in (only or RADARS):
            sd0 = nusc.get("sample_data", sample["data"][ch])
            cands = [sd0]
            for direction in ("prev", "next"):
                sd = sd0
                while sd[direction]:
                    sd = nusc.get("sample_data", sd[direction])
                    if abs(sd["timestamp"] * 1e-6 - t) > self.radar_window:
                        break
                    cands.append(sd)
            want += [(sd, ch) for sd in cands if abs(sd["timestamp"] * 1e-6 - t) <= self.radar_window]
        self._prefetch([sd for sd, _ in want], RadarPointCloud)
        frames = []
        for sd, ch in want:
            fr = self._radar_frame(sd, ch, RadarPointCloud)
            if fr is not None:
                frames.append(fr)
        return frames

    def _prefetch(self, sds, RadarPointCloud):
        """read the radar files that are not cached yet in parallel: opening a file on the raw-data disk takes about 50 ms, a keyframe needs
        10 - 15 of them, and the frames are made from the points in the same order as before (the result does not change)"""
        todo = [sd for sd in sds if sd["token"] not in self._radar_cache and sd["token"] not in self._points]
        if len(todo) < 2:
            return
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            for sd, pts in zip(todo, ex.map(lambda s: RadarPointCloud.from_file(os.path.join(self.root, s["filename"])).points, todo)):
                self._points[sd["token"]] = pts

'''
s = s.replace(old, new, 1)
old2 = "        pts = RadarPointCloud.from_file(os.path.join(self.root, sd[\"filename\"])).points\n"
assert old2 in s
s = s.replace(old2, "        pts = self._points.pop(key, None)\n        if pts is None:\n            pts = RadarPointCloud.from_file(os.path.join(self.root, sd[\"filename\"])).points\n", 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("adapter edited")
