"""scratch: which autogt modules does a training process have loaded after its imports, and which autogt modules do function-level imports of the loaded ones reach later?"""
import os, re, sys
code = "runs/t10_dual_long_s0/code"
sys.path.insert(0, code)
import torch  # noqa
import nuscenes_dataset, losses, train, model  # noqa  the modules a run imports
loaded = sorted(m for m in sys.modules if m == "autogt" or m.startswith("autogt."))
print("loaded after the imports:", loaded)
lazy = set()
for m in loaded:
    f = sys.modules[m].__file__
    if f and f.endswith(".py"):
        for ln in open(f, encoding="utf-8"):
            if re.match(r"\s{4,}(from \.|from autogt|import autogt)", ln):
                lazy.add((os.path.basename(f), ln.strip()[:90]))
print("function-level imports of autogt inside the loaded autogt modules:")
for x in sorted(lazy): print("  ", x)
for f in sorted(os.listdir(code)):
    if f.endswith(".py"):
        for i, ln in enumerate(open(os.path.join(code, f), encoding="utf-8"), 1):
            if re.match(r"\s{4,}(from autogt|import autogt)", ln):
                print("   snapshot", f, i, ln.strip()[:90])
