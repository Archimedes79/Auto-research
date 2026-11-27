"""Pictures from STORED predictions: a frame's raw data and ground truth come from the problem's source, the prediction from the run's stored
result set; no network runs (`Evaluation.draw_frame`). Started by `research report` with the problem's interpreter from the problem folder:

    python -m research.draw <problem> < [{"run": "E1.a_s0", "frame": "val_003/2", "view": "overview", "out": "/abs/path.png"}, ...]

A picture is refused when the ground truth the source gives now is not the one the run was evaluated on (another signature)."""
from __future__ import annotations

import json
import os
import sys

from research.paths import ResearchError, read_json, record_dir


def main(problem: str) -> int:
    from Evaluation import Evaluation, EvaluationError
    items = json.load(sys.stdin)
    ev = Evaluation(project=os.getcwd())
    done = []
    for it in items:
        res = read_json(os.path.join(record_dir(problem, it["run"]), "result.json"))
        try:
            picture = ev.draw_frame(res["evaluation"]["id"], it["frame"], it.get("view", "overview"), it["out"])
        except EvaluationError as e:
            raise ResearchError(f"{it['run']}: {e}") from None
        done.append({"run": it["run"], "frame": picture["frame"], "out": picture["image"], "scores": picture["scores"]})
    json.dump(done, sys.stdout)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except ResearchError as err:
        print(f"refused: {err}", file=sys.stderr)
        sys.exit(1)
