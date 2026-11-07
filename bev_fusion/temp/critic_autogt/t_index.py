import json, os, shutil, types, tempfile
from autogt import cli
from autogt.grid import GridSpec
SIG = "15efbde2d1"
def gen(): return types.SimpleNamespace(signature=lambda: SIG, grid=GridSpec(res=0.5), ss=4)
d = tempfile.mkdtemp(dir="bev_fusion/temp/critic_autogt")
def touch(n): 
    for i in range(n): open(os.path.join(d, f"t{i}.npz"), "w").write("x")
def show(tag):
    i = json.load(open(os.path.join(d, "index.json")))
    print(tag, {k: (len(v) if k == "scenes" else v) for k, v in i.items() if k in ("version", "scenes", "stride", "keyframes", "complete", "gt_version")})
big = [f"scene-{i:04d}" for i in range(850)]
# A: interrupted trainval run (complete False), then a small mini run finishes
touch(300)
cli.write_index(d, gen(), "v1.0-trainval", big, 1, 34149, {}, complete=False); show("A0 trainval provisional")
cli.write_index(d, gen(), "v1.0-mini", big[:10], 1, 404, {}, complete=False); show("A1 mini provisional")
cli.write_index(d, gen(), "v1.0-mini", big[:10], 1, 404, {}); show("A2 mini final   ")
# B: complete trainval stride 1, then the same scenes with stride 2
os.remove(os.path.join(d, "index.json"))
cli.write_index(d, gen(), "v1.0-trainval", big, 1, 34149, {}); show("B0 trainval stride1")
cli.write_index(d, gen(), "v1.0-trainval", big, 2, 17000, {}); show("B1 same scenes stride2")
# C: trainval stride 1 then subset with stride 2 
os.remove(os.path.join(d, "index.json"))
cli.write_index(d, gen(), "v1.0-trainval", big, 1, 34149, {})
cli.write_index(d, gen(), "v1.0-mini", big[:10], 2, 200, {}); show("C  subset stride2 after full stride1")
shutil.rmtree(d)
