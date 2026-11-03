import json
c = json.load(open("training/configs/t17_data.json", encoding="utf-8"))
c["name"] = "t18_fix"
c["_about"] = ("t18 fix (D46 plan, bundle arm): t17_data.json (700 train scenes, the t16 chain and recipe) + the hygiene flags of the Training chat (loss.hygiene_2026_11_02: the weights "
               "w_unknown / w_interpolated applied, unknown velocity out of the moving / static BCE and the velocity losses, splat positions in float32, dead radar_slot branch removed, batch_ego "
               "raising on other dts) + the motion block on its own learning rate (lr_groups motion_block lr 1e-4, warm-up 1000: card experiments/t18_motion_lr). A BUNDLE: a static-EPE gain "
               "is credited to the lr only if no flag touches M's optimisation (card section 'Bundle and attribution'); read against t17.data.")
c.setdefault("loss", {})["hygiene_2026_11_02"] = True
c["lr_groups"] = {"motion_block": {"lr": 1e-4, "warmup": 1000}}
json.dump(c, open("training/configs/t18_fix.json", "w", encoding="utf-8"), indent=1)
open("training/configs/t18_fix.json", "a").write("\n")
print("written", c["loss"], c["lr_groups"], len(c["train_scenes"]))
