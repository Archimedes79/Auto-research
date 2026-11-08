import subprocess
def head(p):
    return subprocess.run(["git", "show", f"HEAD:{p}"], capture_output=True, text=True, check=True, encoding="utf-8").stdout
s = head("bev_fusion/training/configs/README.md")
key = "| `t16_ctl.json` |"
assert s.count(key) == 1
row = ("| `t17_data.json` | experiment t17_more_data (`../../experiments/t17_more_data/`): `t16_full.json` with the 272 train scenes of nuScenes parts 7 - 10 added (700 train / 82 val scenes), caches in `data/cache/all/` "
       "(made by `temp/t16/make_more.py` and `prep_more.sh`: links to the DualBEV shards of all 850 scenes, BEVFusion shards built for the new ones) | written 2026-11-02, queued |\n")
open("bev_fusion/temp/t16/configs_README3.md", "w", encoding="utf-8").write(s.replace(key, row + key))
s = head("bev_fusion/experiments/README.md")
key = "| `t16_full_network/` |"
assert s.count(key) == 1
row = "| `t17_more_data/` | does adding the 272 train scenes of nuScenes parts 7 - 10 to the full network's recipe (t16) improve it? one change, read against `t16.full_s0` on the same 1000 val frames | queued behind the t16 baseline's evaluation (2026-11-02) |\n"
open("bev_fusion/temp/t16/experiments_README3.md", "w", encoding="utf-8").write(s.replace(key, row + key))
print("ok")
