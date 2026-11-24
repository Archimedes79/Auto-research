"""E389 scratch (2026-11-19, D128): prints the card's own words as shell assignments (HYP / PRED / RULE) for the run.py launches of temp/e389/e389_train.sh, so no text is copied by hand.
usage (project folder): eval "$(python3 temp/e389/e389_texts.py)\""""
import os
import shlex
import sys

R = os.environ.get("BFLC_ROOT", ".")
sys.path.insert(0, os.path.join(R, "tools"))
import jarm  # noqa: E402

h, p, r = jarm.card_texts("E389", R)
print(f"HYP={shlex.quote(h)}\nPRED={shlex.quote(p)}\nRULE={shlex.quote(r)}")
