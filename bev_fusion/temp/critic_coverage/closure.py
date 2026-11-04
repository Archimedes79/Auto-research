import ast, os
import autogt
from autogt.nuscenes_gt import SIGNATURE_FILES
from autogt import versions as V
here_ = os.path.dirname(os.path.abspath(autogt.__file__))
todo_, reach_ = ["nuscenes_gt"], set()
while todo_:
    mod_ = todo_.pop()
    if mod_ in reach_: continue
    reach_.add(mod_)
    for n_ in ast.walk(ast.parse(open(os.path.join(here_, mod_ + ".py"), encoding="utf-8").read())):
        if isinstance(n_, ast.ImportFrom) and n_.level == 1:
            todo_ += [n_.module] if n_.module else [a_.name for a_ in n_.names]
print("reach", sorted(reach_))
print("SIGNATURE_FILES", SIGNATURE_FILES)
print("equal:", {m_ + ".py" for m_ in reach_} == set(SIGNATURE_FILES))
loose_ = sorted(set(f_ for f_ in os.listdir(here_) if f_.endswith(".py")) - set(SIGNATURE_FILES) - set(V.NOT_GENERATOR))
print("loose", loose_)
print("NOT_GENERATOR", V.NOT_GENERATOR)
from autogt.nuscenes_gt import NuScenesGT
import hashlib
h = hashlib.sha1()
for name in SIGNATURE_FILES: h.update(open(os.path.join(here_, name), "rb").read())
print("hash of files only (no flags):", h.hexdigest()[:10])
print("versions with signature:", [(r["version"], r["signature"]) for r in V.table()][-3:])
