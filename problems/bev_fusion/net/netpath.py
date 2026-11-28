"""The folder the BEV code works in: <data root>/bev_fusion (RESEARCH_DATA from the research worker, else the machine file). It holds data/cache,
models/pretrained and a link runs -> the problem's run root, so the relative paths of every config stay valid. A script that reads such a path changes into it
(`os.chdir(netpath.SITE)`) in its entry function, never at import time."""
import json
import os


def _site():
    data = os.environ.get("RESEARCH_DATA", "")
    if not data:
        try:
            with open(os.path.expanduser(os.environ.get("RESEARCH_MACHINE") or "~/.research/machine.json"), encoding="utf-8") as f:
                data = json.load(f).get("data", "")
        except (OSError, ValueError):
            data = ""
    return os.path.join(data, "bev_fusion") if data else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SITE = _site()
