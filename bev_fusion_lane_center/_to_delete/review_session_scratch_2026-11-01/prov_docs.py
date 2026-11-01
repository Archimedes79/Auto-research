import sys
def sub(p, old, new):
    s = open(p, encoding="utf-8").read()
    assert s.count(old) == 1, (p, s.count(old))
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("ok", p)
row_old = "`compare` of two cards, `repro <run id>`; design `reproducibility.md` |"
row_new = "`compare` of two cards, `manifest <run id>` (the provenance of a training run from its record and run folder, with the model bundle: the network and loss modules by import closure, hash and copy), `repro <run id>`; design `reproducibility.md` |"
sub("autoresearch/reproducibility.md", "Open: `compare()` must treat an error-only block like `decide.provenance_check` does (as unchecked, exit 2 with a blocker).",
    "`compare()` treats an error-only block like `decide.provenance_check` (a blocker). Training side (Mirko: only the network and the losses form a bundle): `provenance.py manifest <run id>` writes `<run records>/<id>/provenance.json` from the run record and the run folder (GT it trained on, final checkpoint and step, config, seed, run status, code tag) and the model bundle: the files that `model.py` and `losses.py` import (static import closure inside the run's own code snapshot `<run folder>/code`; the training loop and the data loaders are not followed but listed when the network imports them), their hash and a copy in `<run folder>/bundle/`; what the bundle does not carry (repository code such as `autogt`, files that do not parse, dynamic imports, a run that is not final) is listed, not hidden. The launcher does not call it yet (run.py end of run: Training chat); `repro <run id>` reads the manifest.")
import os
for p in sys.argv[1:]:
    sub(p, row_old, row_new)
