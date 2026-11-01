import os
os.chdir("/mnt/c/Auto-research")
p = "autogt/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, ln in enumerate(lines) if ln.startswith("| v7 |"))
lines[i] = ("| v7 | b84e8ac9b2 | coverage (D36 - D38): boxes hide by their height (softer `occluded`), `coverage.py`, per box `share_camera` / `share_radar` / "
            "`share_fusion` and the nuScenes point counts; `camera`, `radar`, `undefined` unchanged. Never built for trainval (superseded by v8 before it finished) |")
lines.insert(i + 1, ("| v8 | c643a7539d | v7 made fast, the same GT (bit-identical on 34 keyframes of four maps): radar files read in parallel, map polygons and "
                     "centre-line points made once, the box shares in one pass, `keyframe_gt` split into methods; **current** |"))
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("ok")
