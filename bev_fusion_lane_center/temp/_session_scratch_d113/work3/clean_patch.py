import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp/_session_scratch_d113/work3/patch_cut.py")
lines = open(p, encoding="utf-8").read().split("\n")
out, skip = [], 0
i = 0
while i < len(lines):
    l = lines[i]
    if l.startswith("sub('''    pieces of a frame (default: the lane GT store of this checkout); `scene`"):
        i += 1
        continue
    if l.startswith("sub('''the point arrays of the lane GT"):
        i += 2
        continue
    if l.startswith("sub('''`gt(token)` -> the point arrays of the lane GT"):
        i += 2
        continue
    if l.startswith("""sub('''    "the sample counts are those of the polylines"""):
        i += 1
        continue
    out.append(l)
    i += 1
open(p, "w", encoding="utf-8").write("\n".join(out))
print(len(lines), len(out))
