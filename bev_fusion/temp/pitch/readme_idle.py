import subprocess
import sys

head = subprocess.run(["git", "show", "HEAD:bev_fusion/evaluation/README.md"], capture_output=True, text=True, encoding="utf-8", cwd="..").stdout
old = "`ps` for trainers / evaluations (nvidia-smi lists no process under WSL2)"
new = "`ps` for a python process running a trainer / evaluation / probe script on the GPU (nvidia-smi lists no process under WSL2; a `run.py` launcher merely queued for a slot is not a user, but starts under a running timing when the slot frees: time through a GPU slot or look at the schedule)"
assert head.count(old) == 1, head.count(old)
open(sys.argv[1], "w", encoding="utf-8").write(head.replace(old, new, 1))
print("ok")
