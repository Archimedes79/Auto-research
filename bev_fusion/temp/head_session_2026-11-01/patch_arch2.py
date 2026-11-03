import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training", "architecture.json")
s = open(p, encoding="utf-8").read()
a = '''    "../autogt/",
    "training/gt_cache_build.py"
   ]'''
assert s.count(a) == 1
s = s.replace(a, '''    "../autogt/",
    "training/gt_cache_build.py",
    "training/gt_scenes.py"
   ]''', 1)
b = '''    "training/train.py::train",
    "training/model.py::check_config",
    "training/feature_cache.py::CachedBEVFusionModel"
   ],'''
assert s.count(b) == 1
s = s.replace(b, '''    "training/train.py::train",
    "training/freeze.py",
    "training/probe.py",
    "training/model.py::check_config",
    "training/feature_cache.py::CachedBEVFusionModel"
   ],''', 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
