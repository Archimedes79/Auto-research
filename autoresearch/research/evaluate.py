"""The evaluation step of a run (started by the worker, inside the run's snapshot, with the problem's interpreter):
the network of the run folder on the evaluation split through `Evaluation` (predictions and counts stay in the run folder), and the record
result.json - or test.json for a run of `research test` - with the numbers, the numbers per scene, the identities and where the counts are.

Only a worker writes these records: without RESEARCH_RECORD (a call by hand) the step refuses; a diagnostic uses `python -m Evaluation`.
A run of `research test` hands `Evaluation` the TEST gate here (`gates.test_refusal`); in every other process a holdout split stays closed."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

from research import gates, spec
from research.paths import ResearchError, read_json, write_json


def snapshot_card(project: str, rec: str) -> Dict[str, Any]:
    """the card of the run's experiment as the snapshot holds it: `project` is the problem folder of the unpacked snapshot, `rec` the run's record
    folder in the checkout. An edit of the card in the checkout while the run goes on changes nothing about what is scored"""
    experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(rec))))
    path = os.path.join(project, "experiments", experiment, "experiment.md")
    with open(path, encoding="utf-8") as f:
        return spec.parse_card(f.read(), path)


def main(problem: str, rid: str, test: bool = False) -> int:
    rec, out, project = os.environ.get("RESEARCH_RECORD", ""), os.environ.get("RESEARCH_RUN_DIR", ""), os.getcwd()
    if not rec or os.environ.get("RESEARCH_RUN") != rid or not out:
        raise ResearchError("the evaluation step writes records and runs only inside a worker; by hand use `python -m Evaluation`")
    p = read_json(os.path.join(project, "problem.json"))
    card = snapshot_card(project, rec)
    if test:
        why = gates.test_refusal(problem, rid)                  # read again here, where the holdout is opened
        if why or os.environ.get("RESEARCH_HOLDOUT_RUN") != rid:
            raise ResearchError(f"TEST refused: {why or 'not a run of research test'}")
        split = p["holdout"][0]
        from Evaluation import defaults
        defaults.set_holdout_gate(gates.test_refusal)           # without it Evaluation keeps every holdout split closed; it asks again at every read
    else:
        split = spec.split_of(card, p)
    frames = split if test else spec.frames_of(card, p)
    from Evaluation import Evaluation
    ev = Evaluation(project=project)
    e = p["evaluate"]
    t0 = time.time()
    done = ev.run(out, frames, metrics=tuple(e["metrics"]), device=e.get("device", "cuda"), workers=0, progress=0)
    nums = ev.show_numbers(done["id"])
    per_scene = ev.split_numbers(done["id"], "scene")
    score = p["score"]["metric"]
    if score not in nums["numbers"]:
        raise ResearchError(f"the score {score} is not among the numbers of the evaluation ({sorted(nums['numbers'])})")
    ident = ev.net_identity(out)
    result = {
        "run": rid, "split": split, "frames": nums["n_frames"], "scenes": len(per_scene),
        "score": {"metric": score, "value": nums["numbers"][score], "better": p["score"]["better"]},
        "numbers": nums["numbers"], "better": nums["better"],
        "per_scene": {sc: row.get(score) for sc, row in sorted(per_scene.items())} if isinstance(per_scene, dict) else {},
        "evaluation": {"id": done["id"], "results": done.get("results"), "frames": frames, "metrics": list(e["metrics"]), "gt": done.get("gt"),
                       "gt_signature": done.get("gt_signature"), "evaluator": done.get("evaluator"), "warnings": nums.get("warnings", []) + done.get("warnings", [])},
        "identity": {"commit": os.environ.get("RESEARCH_COMMIT", ""), **ident},
        "cost": ev.get_cost(done["id"]).get("predict"), "seconds": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(os.path.join(rec, "test.json" if test else "result.json"), result)
    print(json.dumps({"run": rid, "split": split, score: result["score"]["value"], "frames": result["frames"]}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1], sys.argv[2], "--test" in sys.argv[3:]))
    except ResearchError as err:
        print(f"refused: {err}", file=sys.stderr)
        sys.exit(1)
