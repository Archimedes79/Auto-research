"""Where the ground truth lives: <data root>/groundtruth. The data root is RESEARCH_DATA (set by the research worker), else "data" of the machine
file (~/.research/machine.json, or the file RESEARCH_MACHINE names). Only a machine without any machine file (none named, none at the default
place: a fresh clone) gets <repository>/data with the folder name of before 2026-11-27 (<repository>/data/autogt); a machine file that is named or
exists and does not give the data root is an error, never that fallback. Not hashed into any signature: moving the stores changes no key."""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _machine_data():
    """`data` of the machine file; "" only when no file is named and none is at the default place. A file that is named or exists must give the data root: one that is
    missing, cannot be read or lacks `data` raises ValueError with its path (a silent fallback would show an empty store and send hours of ground truth into the repository)"""
    named = os.environ.get("RESEARCH_MACHINE")
    path = os.path.expanduser(named or "~/.research/machine.json")
    if not named and not os.path.exists(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f).get("data")
    except (OSError, ValueError, AttributeError) as e:
        raise ValueError(f"the machine file {path} cannot be read ({type(e).__name__}: {e}): it names the data root of the ground truth. Repair it (`research machine`) or set RESEARCH_DATA") from None
    if not data or not isinstance(data, str):
        raise ValueError(f"the machine file {path} names no data root (`data`), so the place of the ground truth is unknown. Add it (`research machine`) or set RESEARCH_DATA")
    return data


def root():
    """the folder of the ground-truth stores (front stores by signature, lane/, cache/, reports/), resolved at every call"""
    data = os.environ.get("RESEARCH_DATA") or _machine_data()
    return os.path.join(data, "groundtruth") if data else os.path.join(REPO, "data", "autogt")
