"""The evaluation step of a run (started by the worker, inside the run's snapshot, with the problem's interpreter):
the network of the run folder on the evaluation split through `Evaluation` (predictions and counts stay in the run folder), and the record
result.json - or test.json for a run of `research test` - with the numbers, the numbers per scene, the identities and where the counts are.

Only a worker writes these records: without RESEARCH_RECORD (a call by hand) the step refuses; a diagnostic uses `python -m Evaluation`.
A run of `research test` hands `Evaluation` the TEST gate here (`gates.test_refusal`); in every other process a holdout split stays closed."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List

from research import gates, spec
from research.paths import ResearchError, read_json, write_json

SHARED = ("Evaluation", "trainkit", "research", "GtGenerator")     # the packages a run imports besides its problem's own


def outside_snapshot(code: str, problem: str) -> List[str]:
    """the files of the loaded modules of the problem's package and the shared packages that do not lie below `code`, the run's snapshot
    (hard rule 3: a module path that reaches another tree, or an install, would score with code that no commit holds)"""
    root = os.path.realpath(code) + os.sep
    out = []
    for name, mod in sorted(sys.modules.items()):
        if name.split(".")[0] in SHARED + (problem,):
            files = [mod.__file__] if getattr(mod, "__file__", None) else list(getattr(mod, "__path__", []))     # a namespace package: its folders
            out += [f for f in files if not os.path.realpath(f).startswith(root)]
    return out


def refuse_foreign(code: str, problem: str) -> None:
    foreign = outside_snapshot(code, problem)
    if foreign:
        raise ResearchError(f"the run imported code outside its snapshot: {', '.join(foreign[:5])}")


def drop_stored(base: str, done: Dict[str, Any]) -> List[str]:
    """removes what one evaluation stored below `base` (the run's evaluation folder): its counts and the predictions they were made from.
    After a hit of the guard they are not the snapshot's work, and a resumed run would reuse them under the snapshot's identity"""
    gone = []
    for folder, key in (("suite/quant", "id"), ("results", "results")):
        name = str(done.get(key) or "").partition("@")[2]
        path = os.path.join(base, *folder.split("/"), name)
        if name and os.sep not in name and name not in (".", "..") and os.path.isdir(path):
            shutil.rmtree(path)
            gone.append(path)
    return gone


def snapshot_card(project: str, rec: str) -> Dict[str, Any]:
    """the card of the run's experiment as the snapshot holds it: `project` is the problem folder of the unpacked snapshot, `rec` the run's record
    folder in the checkout. An edit of the card in the checkout while the run goes on changes nothing about what is scored"""
    experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(rec))))
    path = os.path.join(project, "experiments", experiment, "experiment.md")
    with open(path, encoding="utf-8") as f:
        return spec.parse_card(f.read(), path)


def main(problem: str, rid: str, test: bool = False) -> int:
    rec, out, project = os.environ.get("RESEARCH_RECORD", ""), os.environ.get("RESEARCH_RUN_DIR", ""), os.getcwd()
    code = os.environ.get("RESEARCH_CODE", "")
    if not rec or os.environ.get("RESEARCH_RUN") != rid or not out or not code:
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
    refuse_foreign(code, problem)                               # the adapter and the core are loaded: asked before anything is predicted or stored
    e = p["evaluate"]
    t0 = time.time()
    done = ev.run(out, frames, metrics=tuple(e["metrics"]), device=e.get("device", "cuda"), workers=0, progress=0)
    try:
        refuse_foreign(code, problem)                           # and again: what the evaluation itself loaded shaped what it stored
    except ResearchError:
        drop_stored(ev.ad.out(ev.resolve(out)), done)
        raise
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
