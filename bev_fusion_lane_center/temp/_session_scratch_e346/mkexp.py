import json, re
src = open("/mnt/c/Auto-research/bev_fusion_lane_center/experiments/E346_one-graph-code-ceiling/experiment.md", encoding="utf-8").read()
m = re.search(r"```json card\n(.*?)\n```", src, re.S)
card = json.loads(m.group(1))
for a in card["arms"]:
    card["arms"][a]["cards"] = [f"temp/e346/out/e346_{a}_n31.json"]
card["evaluate"] = {"split": "val", "subset": "stride29"}; card["band"] = 0.03
open("/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_e346/exp/experiment.md", "w", encoding="utf-8").write("# E346 scratch dry run\n```json card\n" + json.dumps(card, indent=1) + "\n```\n")
print("written")
