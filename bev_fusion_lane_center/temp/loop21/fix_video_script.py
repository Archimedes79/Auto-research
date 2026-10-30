p = "temp/loop21/make_results_video.py"
t = open(p, encoding="utf-8").read()
a = '''        info = infos[int(rows[i])]
        assert int(info["timestamp"]) == int(ts[i]), "image / graph timestamp mismatch"
'''
b = '''        j = int(np.argmin(np.abs(INFO_TS - ts[i])))
        assert abs(INFO_TS[j] - ts[i]) < 1e-3, "no camera info with the timestamp of this frame"
        info = infos[j]
'''
assert a in t
t = t.replace(a, b)
a2 = 'LOCN = {0:'
b2 = 'INFO_TS = np.array([i["timestamp"] for i in infos], dtype=np.float64) / 1e6  # the poses file is ordered differently from the infos: match by timestamp\nLOCN = {0:'
t = t.replace(a2, b2, 1)
t = t.replace("Layer 0: the poses timestamp of every used frame equals the timestamp of the camera info (image and graph belong to the same keyframe).", "Layer 0: every used frame is matched to its camera info by timestamp (within 1 ms; the poses file and the infos are ordered differently), so image and graph belong to the same keyframe.")
open(p, "w", encoding="utf-8").write(t)
