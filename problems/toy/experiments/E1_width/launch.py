"""E1: the four runs of the card through Training (card, snapshot and tag, run record, ledger row), each variant after the control of its seed.

    python3 experiments/E1_width/launch.py [--check]      (from the problem folder; standard library only; --check starts nothing)

Reads the arms, seeds, rule and kill from the card next to this file: nothing of the experiment is written twice.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AR = os.environ.get("AR_ROOT") or os.path.join(os.path.dirname(os.path.dirname(ROOT)), "autoresearch")
sys.path[:0] = [AR, os.path.join(AR, "tools")]

import experiment as card_tool  # noqa: E402
from Training import Training  # noqa: E402

BASE = "training/configs/base.json"


def jobs(card):
    """one job per arm and seed, control first within a seed"""
    control = card["control"]
    letters = [control] + [a for a in card["arms"] if a != control]
    for k, seed in enumerate(card["arms"][control]["seeds"]):
        for letter in letters:
            arm = card["arms"][letter]
            settings = {key: json.loads(value) for key, value in (item.split("=", 1) for item in arm["recipe"].split())}
            settings["seed"] = seed
            job = {"id": arm["run_ids"][k], "config": BASE, "set": settings, "kill": [card["kill"]],
                   "card": {"hyp": f"{card['title']} ({arm['recipe']})", "pred": f"{card['prediction']['value']:+.3f} {card['metric']} over arm {card['prediction']['vs']}",
                            "rule": json.dumps(card["rule"])}}
            if letter != control:
                before = card["arms"][control]["run_ids"][k]
                settings["compare_to"] = before.replace(".", "_")
                job["after"] = [before]
            yield job


def main(argv):
    card = card_tool.read(os.path.join(HERE, "experiment.md"))
    # the data once, before the four jobs: the "prepare" of Training would build one run per job, four jobs would make one store four times
    if subprocess.run(["bash", "tools/prepare.sh", "have", BASE], cwd=ROOT).returncode != 0:
        subprocess.run(["bash", "tools/prepare.sh", "make", BASE], cwd=ROOT, check=True)
    training = Training(ROOT)
    for job in jobs(card):
        if "--check" in argv:
            result = training.check(job)
            print(job["id"], "ok" if result["ok"] else "PROBLEMS " + "; ".join(result["problems"]), "|", result["command"])
        else:
            result = training.run(job)
            print(job["id"], "started" if result["started"] else "not started")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
