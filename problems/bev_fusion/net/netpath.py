"""The folders of the model on the import path (the project imports its modules flat): net/prep, net/model, net/loss. Use it after `net/` is on the path:

    sys.path.insert(1, <bev_fusion>/net); import netpath

The folders are APPENDED, so a pinned run snapshot at the front of the path wins; in a flat snapshot (runs/<name>/code) they do not exist and nothing is added."""
import os
import sys

_NET = os.path.dirname(os.path.abspath(__file__))


def _site():
    """the folder the BEV code works in: <data root>/bev_fusion (RESEARCH_DATA from the research worker, else the machine file); it holds data/cache,
    models/pretrained and a link runs -> the problem's run root, so the relative paths of every config stay valid"""
    import json
    data = os.environ.get("RESEARCH_DATA", "")
    if not data:
        try:
            with open(os.path.expanduser(os.environ.get("RESEARCH_MACHINE") or "~/.research/machine.json"), encoding="utf-8") as f:
                data = json.load(f).get("data", "")
        except (OSError, ValueError):
            data = ""
    return os.path.join(data, "bev_fusion") if data else os.path.dirname(_NET)


SITE = _site()
for _d in ("prep", "model", "loss"):
    _p = os.path.join(_NET, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
