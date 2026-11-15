import sys, itertools
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion/training")
import losses as L, layers as LA
print("DIST_MAX", L.DIST_MAX)
for cl, dn, c1 in itertools.product((False, True), (False, True), (False, True)):
    for tag, spec in (("0.5m", L.layer_spec(cl, dn)), ("1m", L.layer_spec_1m(c1))):
        if not spec:
            continue
        line = [(n, k) for n, kd, k in spec if kd == "line"]
        dist = [(n, k) for n, kd, k in spec if kd == "dist"]
        ok = [a[0] + "_dist" for a in line] == [b[0] for b in dist] and [a[1].replace("@1m", "") + "_dist" + ("@1m" if "@1m" in a[1] else "") for a in line] == [b[1] for b in dist]
        print(f"centrelines={cl} dense={dn} cl1m={c1} level={tag}: line={[a[0] for a in line]} dist={[b[0] for b in dist]} paired={ok}")
# a table with two dist-bearing blocks interleaved with a no-dist polyline
rows = [LA.Layer("a", "polyline", dist=True), LA.Layer("c", "polyline", direction=True), LA.Layer("b", "polyline", dist=True), LA.Layer("e", "polyline", dist=True, direction=True)]
spec = LA.channels(rows)
print(spec)
line = [n for n, kd, k in spec if kd == "line"]; dist = [n for n, kd, k in spec if kd == "dist"]
print("interleaved:", line, dist, [n + "_dist" for n in line] == dist)
