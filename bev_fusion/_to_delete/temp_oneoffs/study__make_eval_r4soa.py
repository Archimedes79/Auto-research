"""One-off (t3 try 1): evaluation copy of t1_r4_soa with the frozen 82 val scenes (same frames as the t2 arms) in its own run folder."""
import json
c = json.load(open("runs/t1_r4_soa/config.json"))
s = json.load(open("data/scenes_parts1_6.json"))
c["name"] = "t3_eval_r4soa"
c["val_scenes"], c["train_scenes"] = s["val"], s["train"]
json.dump(c, open("runs/t3_eval_r4soa/config.json", "w"), indent=1)
print("ok", len(c["val_scenes"]), c["steps"])
