p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()
old = 'fig.suptitle(f"dt = {DT} s; velocity:'
new = 'fig.suptitle(("nuScenes keyframes k-1, k, k+1 (about 0.5 s apart)" if DATA == "nuscenes" else f"dt = {DT} s") + "; velocity:'
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("ok")
