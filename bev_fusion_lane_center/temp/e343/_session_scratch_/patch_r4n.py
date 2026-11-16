import os
R = os.environ["BFLC_ROOT"]
M = os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py")
lines = open(M, encoding="utf-8").read().split("\n")


def find(name):
    idx = [i for i, ln in enumerate(lines) if ln.startswith(f' ("{name}",')]
    assert len(idx) == 1, (name, idx)
    return idx[0]


def setrow(name, old, new):
    lines[find(name)] = f" ({name!r}, {old!r}, {new!r}),".replace("'" + name + "'", '"' + name + '"', 1)


def drop(name):
    del lines[find(name)]


setrow("score_seed B flag unchecked", 'if i["b_made"] else None, split, "B\'s dump", n, exp(run_b))}', 'if i["b_made"] else None, split, "B\'s dump", n, exp(run_b) if i["b_made"] else None)}')
drop("score_seed ctl ends unchecked")
drop("score_seed ctl B unchecked")
setrow("tol REHEARSAL always", "    tol = tol_of(split)\n    ok, worst, where", "    tol = REHEARSAL\n    ok, worst, where")
setrow("score_seed counts always", 'control["counts"] = ctl_counts if control["repro"]["ok"] else None', 'control["counts"] = ctl_counts')
setrow("score_seed repro tol", 'match(numbers_of(ctl_counts), view["numbers"], EXACT)', 'match(numbers_of(ctl_counts), view["numbers"], REHEARSAL)')
setrow("check_seed no exit", '    if ck["seed"] != seed:\n        sys.exit(', '    if False:\n        sys.exit(')
setrow("precheck default without sidecar ok", "        if required or sidecar_provenance(path) is not None:", "        if sidecar_provenance(path) is not None:")
setrow("message interrupted", "made by other code or by an older version of this module", "made by other code")
drop("expect ignored")
setrow("expect short prefix ok", "len(want) >= 16 and digest.startswith(want)", "digest.startswith(want)")
setrow("expect card names none", 'ch = sorted(k for k in set(ref["code_sha256"]) | set(code0) if ref["code_sha256"].get(k) != code0.get(k))', "ch = []")
setrow("precheck seed unchecked flagged", "        if required or sidecar_provenance(path) is not None:", "        if required:")
setrow("expect same card message", "the card's file hashes equal the current ones: that digest is not this card's", "the card is fine")
open(M, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
