"""scratch (D65 step 2, 2026-11-08): training/configs/t20_base_nol4.json = t20_base.json with the stride-4 lane branch (L4) off and nothing else changed.
The source file is read, `model.lane_branch` is set to "" (the ModelConfig default = off), `name` and `_about` change; the diff is asserted. Run from bev_fusion/:  python temp/t22/lane_off/make_config.py"""
import json
import os

SRC, DST = "training/configs/t20_base.json", "training/configs/t20_base_nol4.json"
if os.path.exists(DST):
    raise SystemExit(f"{DST} exists already")
cfg = json.load(open(SRC))
new = json.loads(json.dumps(cfg))
assert new["model"]["lane_branch"] == "res4"
new["model"]["lane_branch"] = ""
new["name"] = "t20_base_nol4"
new["_about"] = ("t20_base.json with the stride-4 lane branch OFF (model.lane_branch \"\"), nothing else changed (D65, Mirko 2026-11-08: \"remove it\", scope chosen: off for new runs; "
                 "t7 gave the branch +0.054 lane divider IoU@0.5 in the old chain, the t20 bundle with it was a TIE against t16_full, and it was never switched off on its own). "
                 "A from-scratch base run: the fine-tune chains t21 / t22 / t23 warm-start from runs/t20_base_s0, whose checkpoint holds the branch, and keep it. "
                 "lane_images stays (the t22 student reads the same image cache). Preview (nothing starts): cd autoresearch && AR_PROJECT=../bev_fusion python3 -m trainsuite start ../bev_fusion/training/configs/t20_base_nol4.json --id t20.base_nol4_s0 --no-card. "
                 "Original _about of t20_base.json: " + str(cfg.get("_about", "")))
top = {k for k in set(cfg) | set(new) if cfg.get(k) != new.get(k)}
mod = {k for k in set(cfg["model"]) | set(new["model"]) if cfg["model"].get(k) != new["model"].get(k)}
assert top == {"name", "_about", "model"} and mod == {"lane_branch"}, (top, mod)
raw = open(SRC, encoding="utf-8").read()
indent = 1 if raw.startswith("{\n ") and not raw.startswith("{\n  ") else 2
json.dump(new, open(DST, "w", encoding="utf-8"), indent=indent)
open(DST, "a").write("\n" if raw.endswith("\n") else "")
print("made", DST, "(indent", indent, "); differs from", SRC, "in name, _about, model.lane_branch")
