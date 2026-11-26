"""The cost numbers of the E1 report, written to cost.json next to this file:
inference (timed now, one network after the other, so the runs of the experiment do not disturb each other) and the experiment (run records and training logs).

    bash tools/py.sh reports/E1/cost.py        (needs the run folders of E1)
"""
import json
import os
import time

from Evaluation.api import Evaluation
from net.prep.shards import ROOT

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["E1.c_s0", "E1.c_s1", "E1.a_s0", "E1.a_s1"]


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return json.load(f)


def main():
    ev = Evaluation(ROOT)
    out = {"measured": time.strftime("%Y-%m-%d %H:%M"), "inference": {}, "experiment": {}}
    for run_id in RUNS:
        run = run_id.replace(".", "_")
        out["inference"][run] = ev.time(f"runs/{run}", 400)
        record = read("records", "runs", run_id, "run.json")
        with open(os.path.join(ROOT, "runs", run, "log.jsonl"), encoding="utf-8") as f:
            steps = [row for row in map(json.loads, f) if "s_per_step" in row]
        times = sorted(row["s_per_step"] for row in steps)
        out["experiment"][run] = {"wall_s": record["duration_s"], "steps": steps[-1]["step"], "median_s_per_step": times[len(times) // 2],
                                  "steps_per_s": 1.0 / times[len(times) // 2], "data_wait_share": None, "gpu_hours": 0.0}
    out["experiment_total_wall_s"] = sum(v["wall_s"] for v in out["experiment"].values())
    with open(os.path.join(HERE, "cost.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for run, v in out["inference"].items():
        print(f"{run}: {v['median_ms']:.4f} ms per frame, {v['parameters']} parameters, load {v['load_1min']}, quiet {v['quiet']}")


if __name__ == "__main__":
    main()
