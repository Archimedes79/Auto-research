import hashlib, os
def files(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", "temp", "_to_delete", "runs", "data", "legacy", "history")]
        for f in fn:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                out.setdefault(f, []).append(p)
    return out
A, B = files("bev_fusion"), files("bev_fusion_lane_center")
C = files("autoresearch")
same_name = sorted(set(A) & set(B))
ident, diff = [], []
for n in same_name:
    for a in A[n]:
        for b in B[n]:
            ha, hb = (hashlib.md5(open(x, "rb").read()).hexdigest() for x in (a, b))
            (ident if ha == hb else diff).append((a, b, os.path.getsize(a)))
print("py files bev_fusion:", sum(len(v) for v in A.values()), " lane:", sum(len(v) for v in B.values()), " autoresearch:", sum(len(v) for v in C.values()))
print("same file name in bev_fusion and lane:", len(same_name), "-> byte-identical pairs:", len(ident), ", different content:", len(diff))
for a, b, sz in ident[:12]:
    print("  identical:", a, "==", b, sz)
for a, b, sz in diff[:8]:
    print("  differs  :", a, "vs", b)
# files of the lane / bev problem that autoresearch also has under the same name
print("same name in autoresearch and a problem folder:", sorted((set(C) & (set(A) | set(B))) - {"selftest.py", "__init__.py"})[:10])
