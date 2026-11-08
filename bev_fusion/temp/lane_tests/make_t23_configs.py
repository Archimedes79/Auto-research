"""scratch: write training/configs/t23_lane_<arm>.json = t21_ctl.json (the equal-budget fine-tune control) + the arm's loss fields and nothing else.
python temp/lane_tests/make_t23_configs.py"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
base = json.load(open(os.path.join(ROOT, "training", "configs", "t21_ctl.json")))
ARMS = {
    "bce": ({"line_bce": 10.0, "line_bce_pos": 3.0, "line_bce_ignore": 1.0},
            "arm C: + a tolerance-aware BCE on the lane / road divider presence (loss.line_bce 10, positives x3, cells within 1 m of a GT line ignored), added to the lines group"),
    "tol0": ({"line_tolerance": 0},
             "arm T: the Dice of the lines without tolerance (loss.line_tolerance 0 instead of 1 cell)"),
}
for arm, (loss, text) in ARMS.items():
    cfg = json.loads(json.dumps(base))
    cfg["_about"] = ("t23 (lane marking, Mirko 2026-11-07): t21_ctl.json (the equal-budget fine-tune of the finished baseline runs/t20_base_s0: 1500 steps, lr 1e-4, window path; its run t21.ctl_s0 is the control) with ONE change: "
                     + text + ". Nothing else differs from t21_ctl.json (seed 0, data, steps, schedule).")
    cfg["name"] = f"t23_lane_{arm}"
    cfg["loss"] = {**cfg["loss"], **loss}
    out = os.path.join(ROOT, "training", "configs", f"t23_lane_{arm}.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")
    print("wrote", out)
a, b = (json.load(open(os.path.join(ROOT, "training", "configs", f"t23_lane_{x}.json"))) for x in ("bce", "tol0"))
for k in set(base) | set(a):
    if base.get(k) != a.get(k) and k not in ("_about", "name"):
        print("differs from t21_ctl (bce):", k, base.get(k), "->", a.get(k))
