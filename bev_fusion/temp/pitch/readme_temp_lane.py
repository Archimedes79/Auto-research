"""lane temp/README.md: the three loop rows (19, 20, 21) become one row 'filed'. Run from bev_fusion_lane_center (WSL python3)."""
p = "temp/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
out = [l for l in lines if not l.startswith(("| `loop19/`", "| `loop20/`", "| `loop21/`"))]
i = [k for k, l in enumerate(out) if l.startswith("| `STRUCTURE.md`")][0]
out.insert(i, "| `loopNN/` | empty now: lane loops 19 - 22 were filed on 2026-11-01 (tags `lcr/loopNN-scratch` hold the scripts, the files are in `_to_delete/temp_loopNN/`); the next loop writes here |")
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out))
print("rows removed:", len(lines) - len(out) + 1)
