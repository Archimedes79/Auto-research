import os
import time
p = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval/sample_data.json")
print(os.path.realpath(p))
t = time.time()
try:
    s = open(p).read()
    print("text read ok", len(s), round(time.time() - t, 1))
except OSError as e:
    print("text read failed", e)
t = time.time()
with open(p, "rb") as f:
    b = f.read()
print("binary read ok", len(b), round(time.time() - t, 1))
