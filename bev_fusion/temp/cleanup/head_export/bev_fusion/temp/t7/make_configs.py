"""t7 scratch: the four arm configs = t2_ref_s<seed> + the lane branch (only change) -> training/configs/t7_<arm>_s<seed>.json."""
import json

for seed in (0, 1):
    base = json.load(open(f"training/configs/t2_ref_s{seed}.json"))
    assert base["seed"] == seed
    for arm in ("res4", "rgb"):
        cfg = dict(base, name=f"t7_{arm}_s{seed}", lane_images="data/cache/img_front")
        cfg["model"] = dict(base["model"], lane_branch=arm)
        keys = list(base)                                    # keep the reference's key order, new keys after "radar_cache"
        order = keys[:keys.index("radar_cache") + 1] + ["lane_images"] + keys[keys.index("radar_cache") + 1:]
        with open(f"training/configs/{cfg['name']}.json", "w", newline="\n") as f:
            json.dump({k: cfg[k] for k in order}, f, indent=1)
            f.write("\n")
        print(cfg["name"])
