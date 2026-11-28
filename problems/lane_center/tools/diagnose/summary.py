"""The verdict of a lane experiment and the stage numbers of its runs (and of the control's) in one table:
`python3 -m research py lane_center -m lane_center.tools.diagnose.summary E<n>` (from autoresearch/)."""
import glob
import json
import os
import sys

from lane_center.training import bflc_paths as BP

PROBLEM = BP.ROOT

METRICS = ("graph.topo_F1", "graph.junction_F1", "graph.F1@1.0", "graph.lanes_found", "graph.dup_frac", "graph.vert100_dp")


def main(exp):
    folder = glob.glob(os.path.join(PROBLEM, "experiments", f"{exp}_*"))[0]
    verdict = json.load(open(os.path.join(folder, "verdict.json")))
    print(f"{exp}: " + "; ".join(f"{a}: {v['verdict']} ({v.get('reason', '')})" for a, v in verdict["arms"].items()))
    for arm, v in verdict["arms"].items():
        if v.get("guards_broken"):
            print(f"  {arm} guards broken: {v['guards_broken']}")
    card = open(os.path.join(folder, "experiment.md"), encoding="utf-8").read()
    runs = sorted(glob.glob(os.path.join(folder, "runs", "*", "result.json")))
    control = json.loads(card.split("```json card")[1].split("```")[0])
    ref = control["arms"].get(control.get("control", ""), {}).get("from")
    if ref:
        other, arm = ref.split(".")
        runs = sorted(glob.glob(os.path.join(PROBLEM, "experiments", f"{other}_*", "runs", f"{other}.{arm}_*", "result.json"))) + runs
    head = ["run"] + [m.split(".")[1] for m in METRICS] + ["forks per frame", "forks with a jump"]
    print(" | ".join(head))
    for r in runs:
        res = json.load(open(r))
        name = res["run"]
        diag = os.path.join(BP.RUNS, "diagnose", name, "diagnose.json")
        d = json.load(open(diag)) if os.path.exists(diag) else None
        extra = [f"{d['forks']['per_frame']:.1f}", f"{d['forks']['share_with_a_jump']:.0%}"] if d else ["-", "-"]
        print(" | ".join([name] + [f"{res['numbers'][m]:.4f}" for m in METRICS] + extra))


if __name__ == "__main__":
    main(sys.argv[1])
