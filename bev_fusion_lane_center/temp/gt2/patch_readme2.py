p = "training/README.md"
s = open(p, encoding="utf-8").read()
a = "(feature extraction, ground truth, poses, priors, splits; the b456 extension: `make_b456_tokens.py`, `run_b456_extract.sh`)"
assert a in s
s = s.replace(a, a[:-1] + "; `junction_paint.py`: E285, the stop lines and pedestrian crossings of the map expansion in the lane frame, one sidecar per TRAIN / VAL keyframe in `<feature dir>/paint/`)", 1)
b = "`qnode_pilot.py` (E282:"
assert b in s
s = s.replace(b, "`jmaps_train.py` (E286: the junction maps head on the pyramid of a baseline run, frozen or joint), " + b, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
