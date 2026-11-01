import re
# training/README.md: the gt_bev row becomes a shim row
p = "training/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
for i, l in enumerate(lines):
    if l.startswith("| `gt_bev.py` |"):
        lines[i] = ("| `gt_bev.py` | compatibility shim (since 2026-10-30): the GT generator lives in the standalone suite `../autogt/` "
                    "(static layers, objects, masks, radar-corrected velocity / flow in m/s, training samples, build / check / view / video "
                    "commands); this file re-exports NuScenesGT, Raster, AREAS / LINES / OBJECTS, cell_centres, occlusion, pose_matrix, "
                    "category_group | `python -m autogt selftest` (from the bev_fusion folder) |")
open(p, "w", encoding="utf-8").write("\n".join(lines))
# bev_fusion/README.md: a row for autogt after the training row
p = "README.md"
lines = open(p, encoding="utf-8").read().split("\n")
for i, l in enumerate(lines):
    if l.startswith("| `training/` |"):
        lines.insert(i + 1, "| `autogt/` | the standalone autoGT suite: automatic GT from nuScenes (static BEV layers, objects, masks, radar-corrected velocity / flow field in m/s), with its own build, check, view and video commands and self-tests - see `autogt/README.md` |")
        break
open(p, "w", encoding="utf-8").write("\n".join(lines))
# plan.md: note at the top of 9.5
p = "plan.md"
s = open(p, encoding="utf-8").read()
marker = "### 9.5 Built and checked: GT per keyframe and training samples (2026-10-25, `training/gt_bev.py`)\n"
note = marker + ("- **2026-10-30: the GT generator is now the standalone suite `autogt/`** (`training/gt_bev.py` is a shim; start with `autogt/README.md`). "
                 "The one-off scripts named below (`check_gt_*`, `debug_*`, `view_gt_*`, `view_training_samples`, `check_map_alignment_locations`, ...) "
                 "were folded into its `check`, `view` and `video` commands and removed; they are in git history (commit 2226cae and before).\n")
assert marker in s
s = s.replace(marker, note, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
