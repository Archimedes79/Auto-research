import json
import os

W = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(W, "step0_full.json")))
b = json.load(open(os.path.join(W, "pot_full2.json")))
worst = 0.0
n = 0


def cmp(x, y, tag):
    global worst, n
    d = abs(float(x) - float(y))
    worst = max(worst, d)
    n += 1
    if d > 0:
        print("DIFF", tag, x, y, d)


cmp(a["samples_per_frame"], b["samples_per_frame"], "samples_per_frame")
for k, v in a["classes"].items():
    for kk, vv in v.items():
        cmp(vv, b["classes"][k][kk], f"{k}.{kk}")
for i, bn in enumerate(a["by_bin"]):
    for kk, vv in bn.items():
        cmp(vv, b["by_bin"][i][kk], f"bin{i}.{kk}")
print(f"{n} numbers compared, largest absolute difference {worst:.3e}")
