import os
R = os.environ["BFLC_ROOT"]
src = open(os.path.join(R, "evaluation", "jtest_eval.py"), encoding="utf-8").read().split("\n")
M = os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py")
lines = open(M, encoding="utf-8").read().split("\n")


def setrow(name, old, new):
    idx = [i for i, ln in enumerate(lines) if ln.startswith(f' ("{name}",')]
    assert len(idx) == 1, (name, idx)
    lines[idx[0]] = f" ({name!r}, {old!r}, {new!r}),".replace("'" + name + "'", '"' + name + '"', 1)


setrow("is_protected no realpath", "real = os.path.realpath(path).casefold()\n    return any", "real = os.path.abspath(path).casefold()\n    return any")
setrow("is_protected no casefold", "real = os.path.realpath(path).casefold()\n    return any", "real = os.path.realpath(path)\n    return any")
cp = "    elif expect is not None:\n        check_provenance(path, expect, what, required=make is not None)"
setrow("get_dump no provenance", cp, cp.replace("elif expect is not None:", "elif False:"))
setrow("get_dump flagged unchecked", cp, cp.replace("elif expect is not None:", "elif expect is not None and make is not None:"))
setrow("get_dump default no sidecar ok", "required=make is not None)", "required=False)")
setrow("provenance missing sidecar ok", '        if required:\n            sys.exit(f"{what}: {path} has no provenance', '        if False:\n            sys.exit(f"{what}: {path} has no provenance')
b_prov = src[[i for i, ln in enumerate(src) if ln.startswith("    prov = provenance(run, device)") and "STARTS" in ln][0]]
e_prov = src[[i for i, ln in enumerate(src) if ln.startswith("    prov = provenance(run, device)") and "as in dump_b" in ln][0]]
setrow("dump_b overwrites", "    refuse_existing(out)\n" + b_prov, b_prov)
setrow("dump_ends overwrites", "    refuse_existing(out)\n" + e_prov, e_prov)
setrow("code no parent packages", ' + [p for p in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if isfile(p)]', "")
open(M, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
