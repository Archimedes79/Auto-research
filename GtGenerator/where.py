"""Where the ground truth lives: <data root>/groundtruth. The data root is RESEARCH_DATA (set by the research worker), else "data" of the machine
file (~/.research/machine.json, or the file RESEARCH_MACHINE names), else <repository>/data with the folder name of before 2026-11-27
(<repository>/data/autogt). Not hashed into any signature: moving the stores changes no key."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _machine_data():
    path = os.path.expanduser(os.environ.get("RESEARCH_MACHINE") or "~/.research/machine.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("data", "")
    except (OSError, ValueError, AttributeError):
        return ""


def root():
    """the folder of the ground-truth stores (front stores by signature, lane/, cache/, examples/, reports/)"""
    data = os.environ.get("RESEARCH_DATA") or _machine_data()
    return os.path.join(data, "groundtruth") if data else os.path.join(REPO, "data", "autogt")
