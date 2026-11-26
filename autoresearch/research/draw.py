"""Pictures from STORED predictions: a frame's raw data and ground truth come from the problem's source, the prediction from the run's stored
result set; no network runs. Started by `research report` with the problem's interpreter from the problem folder:

    python -m research.draw <problem> < [{"run": "E1.a_s0", "frame": "val_003/2", "view": "overview", "out": "/abs/path.png"}, ...]

A picture is refused when the ground truth the source gives now is not the one the run was evaluated on (another signature)."""
from __future__ import annotations

import json
import os
import sys

from research.paths import ResearchError, read_json, record_dir, run_dir


def main(problem: str) -> int:
    from Evaluation import Evaluation
    from Evaluation.contracts import FrameRef, ViewContext
    from Evaluation.render import figure
    items = json.load(sys.stdin)
    ev = Evaluation(project=os.getcwd())
    done = []
    for it in items:
        res = read_json(os.path.join(record_dir(problem, it["run"]), "result.json"))
        info, store = ev._open_store(res["evaluation"]["id"])
        rs = ev._predictions(info)
        if rs is None:
            raise ResearchError(f"the stored predictions of {it['run']} are gone: evaluate it again before drawing")
        parsed = FrameRef.parse(it["frame"])
        if not parsed:
            raise ResearchError(f"frame {it['frame']!r}: <scene>/<index>")
        net = run_dir(problem, it["run"])
        source = ev._source(net, res["split"], 0, "", True)
        try:
            now = source.describe().get("gt_signature")
            if info.get("gt") and now and str(now) not in json.dumps(info["gt"]):
                raise ResearchError(f"{it['run']} was evaluated on ground truth {info['gt']}, the source gives {now} now: the picture would not show what was scored")
            ref = next((r for r in source.frames(parsed[0]) if r.index == parsed[1]), None)
            if ref is None:
                raise ResearchError(f"no frame {it['frame']} in {res['split']}")
            sample, pred = source.load(ref), rs.load(ref)
            task = ev._task(net, tuple(res["evaluation"]["metrics"]))
            scores = {}
            for m in task.metrics():
                scores.update({f"{m.name}.{k}": float(v) for k, v in m.score(m.frame(sample, pred)).items()})
            page = task.views()[it.get("view", "overview")].render(sample, pred, ViewContext(scores=scores))
            os.makedirs(os.path.dirname(os.path.abspath(it["out"])), exist_ok=True)
            figure.render_page(page, it["out"])
            done.append({"run": it["run"], "frame": ref.label, "out": it["out"], "scores": scores})
        finally:
            if hasattr(ev.ad, "release"):
                ev.ad.release(source)
    json.dump(done, sys.stdout)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except ResearchError as err:
        print(f"refused: {err}", file=sys.stderr)
        sys.exit(1)
