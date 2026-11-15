"""scratch: the 43 scene names of the common frames (keys of the converted CRN record) -> temp/t26/scenes43.txt, and the variant folder of the base evaluation"""
import json, os, pickle
rec = pickle.load(open(os.path.expanduser("~/work/CRN/outputs/t26_crn_lead/oracle/rec.pkl"), "rb"))
names = sorted(rec["det"].keys())
open("temp/t26/scenes43.txt", "w").write("\n".join(names) + "\n")
print(len(names), "scenes;", sum(len(v[0.0]) for v in rec["det"].values()), "frames; meta keys:", sorted(rec["meta"]))
cfg = json.load(open("runs/t20_base_s0/config.json"))
cfg["name"] = "t26_base_eval"
os.makedirs("temp/t26/base_eval", exist_ok=True)
json.dump(cfg, open("temp/t26/base_eval/config.json", "w"), indent=1)
print("variant folder temp/t26/base_eval; init_from:", cfg.get("init_from"), "gt_cache:", cfg.get("gt_cache"))
