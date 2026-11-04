import subprocess, sys
head = subprocess.run(["git", "show", "HEAD:bev_fusion/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8", cwd="..").stdout
old = "| `bev_views.py` | the pictures as `Page`s: `overview` (camera with the projected GT boxes and map lines), `outputs`, `stages`, `dashboard` |"
assert head.count(old) == 1, head.count(old)
new = ("| `bev_views.py` | the pictures as `Page`s: `overview` (camera with the projected GT boxes and map lines), `outputs`, `stages`, `dashboard`; colour scheme of the bird's-eye maps (Mirko, 2026-11-03): "
       "black background, dark-grey road with a black boundary, white lane lines, light-grey crossings, calmer green walkways, dark-red objects and the same red translucent outside the observed area "
       "(camera x radar x not hidden), GT outline red, dashed GT outline white, predicted boxes amber (the error maps keep grey / red / blue / cream) |")
open(sys.argv[1], "w", encoding="utf-8").write(head.replace(old, new, 1))
print("ok")
