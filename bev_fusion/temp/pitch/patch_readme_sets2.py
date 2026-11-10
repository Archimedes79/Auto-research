import sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("(`--real` runs a named set against the run's own)", "(`--sets` runs a named set against the run's own, CPU, by hand)")
rep("Not offered on a named set (they are the run's own evaluation): `--all`, `generalisation`, `report`, `memory`. Whether the features and GT of a set's scenes exist is the data's business, as for a training: a scene without cached features cannot be run (`bev_suite.py RUN --split SET list` shows what the run can read).",
    "Pictures, videos, the module probe and `quant` stores of a named set go to `eval/suite_<split>_<part>/` and `eval/modules_<split>_<part>/`, not into the run's plain folders; the prior baseline (`--prior`) of a set has its own store (`prior_<split>_<part>500_n...`, built from the split's train scenes). The set's scene hash is part of the store identity, so a split file edited under the same name is refused on resume. Not offered on a named set (they write the run's plain files): `--all`, `generalisation`, `report`, `memory`, `--time` / `time` / `timing`. A set whose scenes the run's data cannot read (not in the official part of nuScenes, no front-camera files) is refused with the names, never evaluated with fewer scenes; whether the cached features and GT of its scenes exist is the data's business as for a training, a scene without them stops at its first frame (`bev_suite.py RUN --split SET list` shows what the run can read). A split file must not be named `val`, `train`, `test` or start with `warm` / `ctl` (`bev_sets.py` refuses it: the bare names, TEST and the output tags would read alike).")
open(path, "w", encoding="utf-8").write(s)
print("ok")
