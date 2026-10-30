p = "temp/loop21/make_results_video.py"
t = open(p, encoding="utf-8").read()
t = t.replace('assert abs(INFO_TS[j] - ts[i]) < 1e-3, "no camera info with the timestamp of this frame"', 'assert abs(INFO_TS[j] - ts[i]) < 0.06, "no camera info within 60 ms of this frame"  # poses carry a sensor timestamp 33 - 42 ms off the sample timestamp; keyframes are 0.5 s apart')
t = t.replace('LOCN = {0:', 'NEAR = np.array([int(np.argmin(np.abs(INFO_TS - x))) for x in pz["ts"]])\nassert len(set(NEAR.tolist())) == len(NEAR), "the nearest-timestamp match is not one-to-one"\nLOCN = {0:', 1)
t = t.replace("(within 1 ms;", "(nearest, within 60 ms, one-to-one on all 1706 frames;")
open(p, "w", encoding="utf-8").write(t)
