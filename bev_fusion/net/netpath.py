"""The folders of the model on the import path (the project imports its modules flat): net/prep, net/model, net/loss. Use it after `net/` is on the path:

    sys.path.insert(1, <bev_fusion>/net); import netpath

The folders are APPENDED, so a pinned run snapshot at the front of the path wins; in a flat snapshot (runs/<name>/code) they do not exist and nothing is added."""
import os
import sys

_NET = os.path.dirname(os.path.abspath(__file__))
for _d in ("prep", "model", "loss"):
    _p = os.path.join(_NET, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
