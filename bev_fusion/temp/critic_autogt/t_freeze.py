import sys, os
sys.path.insert(0, os.path.abspath("bev_fusion/temp/critic_autogt/pkg"))
import autogt
from autogt import suite
print("package used:", os.path.dirname(autogt.__file__))
gen = suite.AutoGT.__mro__[1]   # NuScenesGT (no devkit needed for signature)
import types
s0 = autogt.current_signature(); v0 = autogt.current_version()
# a "running training": already holds a signature from the object API?  signature() of a constructed generator is frozen; load()/missing() are not
with open(os.path.join(os.path.dirname(autogt.__file__), "layers.py"), "a") as f: f.write("\n# edit while a job runs\n")
s1 = autogt.current_signature(); v1 = autogt.current_version()
print("before edit:", s0, v0, "| after an edit of layers.py:", s1, v1)
print("load() looks into:", os.path.join(suite.DATA, s1))
print("missing(['x']) now uses folder", s1)
