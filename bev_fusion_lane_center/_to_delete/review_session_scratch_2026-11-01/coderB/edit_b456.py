"""D24: the b456 builders moved from temp/loop19/ to training/data/prep/"""
import os
R = os.environ["BFLC_ROOT"]
def sub(path, old, new, n=1):
    p = os.path.join(R, path)
    t = open(p, encoding="utf-8").read()
    assert t.count(old) == n, (path, old[:60], t.count(old))
    open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))
    print("ok", path)
sub("training/data/prep/run_b456_extract.sh", "Usage: wsl bash temp/loop19/run_b456_extract.sh", "Usage: wsl bash training/data/prep/run_b456_extract.sh (moved from temp/loop19/ on 2026-10-31, D24)")
sub("training/data/prep/run_b456_extract.sh", 'cd "$(dirname "$0")/../.." || exit 1                      # the problem folder', 'cd "$(dirname "$0")/../../.." || exit 1                   # the problem folder')
sub("training/data/prep/make_b456_tokens.py", "    wsl bash tools/env/_py14.sh temp/loop19/make_b456_tokens.py", "    wsl bash tools/env/_py14.sh make_b456_tokens.py      (training/data/prep/, moved from temp/loop19/ on 2026-10-31, D24)")
sub("data/rebuild.json", '"temp/loop19/make_b456_tokens.py"', '"training/data/prep/make_b456_tokens.py"')
sub("data/rebuild.json", '"temp/loop19/run_b456_extract.sh"', '"training/data/prep/run_b456_extract.sh"')
sub("training/data/prep/gcar_tools.py", "token lists from temp/loop19/make_b456_tokens.py", "token lists from training/data/prep/make_b456_tokens.py")
sub("temp/README.md", "| `loop19/` | scratch of loop 19 (filed); not yet cleaned: `data/rebuild.json` names `loop19/make_b456_tokens.py` and `loop19/run_b456_extract.sh` as builders of the b456 split and features (move them to `training/data/prep/` first) |",
    "| `loop19/` | scratch of loop 19 (filed); its b456 builders `make_b456_tokens.py` and `run_b456_extract.sh` moved to `training/data/prep/` on 2026-10-31 (D24, `data/rebuild.json` updated); `run_b456_gt.sh` (b456 ground truth, E235) is still here and not yet in `data/rebuild.json` |")
sub("training/README.md", "**`prep/`** = builders of the cached data (feature extraction, ground truth, poses, priors, splits) |",
    "**`prep/`** = builders of the cached data (feature extraction, ground truth, poses, priors, splits; the b456 extension: `make_b456_tokens.py`, `run_b456_extract.sh`) |")
