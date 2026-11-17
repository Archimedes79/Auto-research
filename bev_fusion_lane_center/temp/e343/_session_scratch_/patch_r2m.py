import os
p = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(p, encoding="utf-8").read()
pairs = [
    ('''its provenance (run, device, sha256 of the checkpoint, digest of the imported lane code): an existing one is reused only under the same provenance, else refused with a message (nothing is deleted: move it away). Dumps named by a flag (E321's, the stored VAL
dumps) are exempt: validated, and E321's also by hash.''', '''its provenance (run, device, sha256 of the checkpoint, digest and file list of the imported lane code): an existing one is reused only under the same provenance, else refused with a message (nothing is deleted: move it away). A dump named by a flag
is exempt when it has no sidecar (E321's, the stored VAL dumps: validated, E321's also by hash) and checked like the others when this module made it.'''),
    ('''a smoke run (`--max N`, never on TEST) writes under `temp/e343/<split>_max<N>/` only.''', '''a smoke run (`--max N`, never on TEST) writes under `temp/e343/<split>_max<N>/` only (cards in `cards_<system>/`).'''),
]
for old, new in pairs:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
