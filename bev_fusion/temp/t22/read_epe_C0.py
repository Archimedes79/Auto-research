"""scratch (t22 gate C0): the two EPE rows of rule C0 (and a few neighbours) from the metrics.json of the five runs, point estimates."""
import json
import os

RUNS = [("control t21_pair4_s0", "runs/t21_pair4_s0"),
        ("stand-in current", "runs/t22_standin_pair4"),
        ("stand-in ALL", "runs/t22_standin_all_pair4"),
        ("B (current)", "runs/t22_B_s0"),
        ("B with ALL", "runs/t22_B_all")]
ROWS = ["vel/dt+0.00/cells/static/epe", "vel/dt+0.00/centre/field/moving/epe", "vel/dt+0.00/centre/field/all/epe",
        "vel/dt+0.00/cells/vehicle/epe", "vel/dt+0.00/cells/vehicle_moving/epe", "curve/dt+0.00/velocity/moving/epe"]


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[p + k] = v
    return out


m = {}
for name, run in RUNS:
    d = flat(json.load(open(os.path.join(run, "eval", "metrics.json")))["metrics"])
    m[name] = d
print(f"{'row':44s}" + "".join(f"{n:>22s}" for n, _ in RUNS))
for k in ROWS:
    print(f"{k:44s}" + "".join(f"{m[n].get(k, float('nan')):22.4f}" for n, _ in RUNS))
