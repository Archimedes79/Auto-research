import subprocess
old = "for the snapped read-out of E312 and the E320 PRIMARY read-out (`score --seed`, `selftest`;"
new = "for six read-outs (E312 snapped and snap-nogeom, E320 own keypoints and PRIMARY, a third run such as E316; `score --seed [--l-run]`, `selftest`;"
head = subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8", cwd="..").stdout
assert head.count(old) == 1
open("temp/e324/README_blob2.md", "w", encoding="utf-8").write(head.replace(old, new))
print("ok")
