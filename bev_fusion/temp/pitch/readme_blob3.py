"""README blob for the commit: HEAD's README + my edits only (the working copy holds other chats' uncommitted lines)"""
import re, subprocess
out = "temp/pitch/README_blob3.md"
head = subprocess.run(["git", "show", "HEAD:bev_fusion/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8", cwd="..").stdout
open(out, "w", encoding="utf-8").write(head)
# step 1: patch_readme.py on the blob
src = open("temp/pitch/patch_readme.py", encoding="utf-8").read().replace('p = "evaluation/README.md"', f'p = "{out}"')
exec(src)
s = open(out, encoding="utf-8").read()
s = s.replace("(about 10 min CPU, on demand) | all 16 caught |", "(about 10 min CPU, on demand) | all 16 caught |")
s = re.sub(r"(metrics `seg` \(SegMeter counts of the masks fov / seen / out / obj per frame\), )(`det` and `detr`)", r"\1`curve` (`--curves`), `detf` / `detfp` / `cov` (`--fusion`, `bev_fusion_area.py`), \2", s)
s = s.replace("180 numbers incl. the role-swapped rows", "182 numbers incl. the role-swapped rows")
open(out, "w", encoding="utf-8").write(s)
# step 3: patch_rest.py README pairs only
src = open("temp/pitch/patch_rest.py", encoding="utf-8").read()
start = src.index('edit("evaluation/README.md"')
src2 = src[:src.index('edit("evaluation/bev_annotation_pts.py"')] + src[start:]
src2 = src2.replace('edit("evaluation/README.md"', f'edit("{out}"')
exec(src2)
print(open(out, encoding="utf-8").read().count("bev_fusion_area"), "mentions;", "189 numbers" in open(out, encoding="utf-8").read(), "neutral rule off" in open(out, encoding="utf-8").read())
