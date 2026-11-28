"""Where the BEV code finds its data, from the data root of the machine (RESEARCH_DATA from the research worker, else `data` of the machine file):

    SITE                  <data root>/bev_fusion, the folder the code works in. It holds data/cache, models/pretrained and a link runs -> the problem's run
                          root, so the relative paths of every config stay valid. A script that reads such a path changes into it (`os.chdir(netpath.SITE)`)
                          in its entry function, never at import time.
    nuscenes_root(mini)   the raw nuScenes data as the code reads it (datasets/nuscenes/README.md): the view <data root>/nuscenes (trainval) or
                          <data root>/nuscenes_mini; where that view is not there, the link of older set-ups to the same folder, ~/data/nuscenes_full or
                          ~/data/nuscenes (what the stored configs name as data_root), and one printed line says so.
"""
import json
import os

VIEWS = {False: ("nuscenes", "~/data/nuscenes_full"), True: ("nuscenes_mini", "~/data/nuscenes")}      # mini? -> (view of the data root, the older link)


def _data_root():
    data = os.environ.get("RESEARCH_DATA", "")
    if not data:
        try:
            with open(os.path.expanduser(os.environ.get("RESEARCH_MACHINE") or "~/.research/machine.json"), encoding="utf-8") as f:
                data = json.load(f).get("data", "")
        except (OSError, ValueError):
            data = ""
    return data


def _site():
    data = _data_root()
    return os.path.join(data, "bev_fusion") if data else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_SAID = set()                                                    # the fallbacks already reported (one line per root and process)


def nuscenes_root(mini=False):
    view, link = VIEWS[bool(mini)]
    data = _data_root()
    path = os.path.join(data, view)
    if data and os.path.isdir(path):
        return path
    old = os.path.expanduser(link)
    if old not in _SAID:
        _SAID.add(old)
        print(f"nuScenes: {'no view ' + path if data else 'no data root'}; reading {old}", flush=True)
    return old


SITE = _site()
