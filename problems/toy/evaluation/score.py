"""The evaluation step of one arm and seed: predict the val frames (the Runner), score them, write the metric card at its fixed path
(results/cards/<run>/val.json), add the paired comparison against the control, two pictures, the worst frames and the timing.

    bash tools/py.sh evaluation/score.py <composed config.json>

A card is a record only when its numbers come from a run with a code commit (AR_COMMIT, set by run.py). Called by hand, or in a run without git, everything goes
to temp/cards/<run>/: a diagnostic never adds to the records and never replaces a card that a verdict used.
"""
import json
import os
import sys

from Evaluation.api import Evaluation, EvaluationError
from net.prep.shards import ROOT


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, default=str)
        f.write("\n")
    os.replace(path + ".tmp", path)


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def card_path(run):
    """where this call writes the card of `run`"""
    return os.path.join(ROOT, *(("results",) if os.environ.get("AR_COMMIT") else ("temp",)), "cards", run, "val.json")


def control_card(run):
    """the card a comparison reads: the one this kind of call would write, else the record"""
    own = card_path(run)
    return own if os.path.isfile(own) else os.path.join(ROOT, "results", "cards", run, "val.json")


def main(argv):
    cfg = read(argv[0])
    run = cfg["name"]
    ev = Evaluation(ROOT)
    done = ev.run(f"runs/{run}", "val", metrics=("seg",), device="cpu", workers=0, progress=0)
    for warning in done.get("warnings", []):
        print("warning:", warning)
    os.makedirs(os.path.dirname(card_path(run)), exist_ok=True)
    ev.make_card(done["id"], out=card_path(run))
    card = read(card_path(run))
    paired = None
    if cfg.get("compare_to"):                                   # the paired scene bootstrap, this run minus the control (the same frames, truth and evaluator)
        control = read(control_card(cfg["compare_to"]))
        try:
            paired = ev.compare(control["info"]["eval_id"], done["id"])
        except EvaluationError as e:
            if os.environ.get("AR_COMMIT"):                     # a run that makes a record needs its interval: fail
                raise
            print(f"not compared: {e}\n(score the control by hand first: its card then comes from the same evaluator)")
    if paired:
        m = paired["metrics"]["seg.iou"]
        card["compare"] = {"name": control["name"], "metrics": control["metrics"], "paired": {"iou": {k: m[k] for k in ("diff", "lo", "hi")}},
                           "n_scenes": paired["n_scenes"], "warnings": paired["warnings"]}
        write(card_path(run), card)
    frames = ev.split_numbers(done["id"], "frame")
    worst = sorted(frames, key=lambda label: frames[label].get("seg.iou", 1.0))[:3]
    pictures = [p for label in ("val_000/0", worst[0]) for p in ev.draw_frame(done["id"], label)["images"].values()] if worst else []
    write(os.path.join(os.path.dirname(card_path(run)), "worst.json"), {"frames": [{"frame": f, "iou": frames[f].get("seg.iou")} for f in worst],
                                                                        "pictures": [os.path.relpath(str(p), ROOT) for p in pictures]})
    write(os.path.join(os.path.dirname(card_path(run)), "timing.json"), ev.time(f"runs/{run}", 200))
    line = f"{run}: iou {card['metrics']['iou']:.4f} on {card['frames']} val frames"
    if "compare" in card:
        p = card["compare"]["paired"]["iou"]
        line += f"; against {card['compare']['name']}: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
