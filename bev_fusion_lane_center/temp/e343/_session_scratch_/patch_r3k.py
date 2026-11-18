import os
R = os.environ["BFLC_ROOT"]
src = open(os.path.join(R, "evaluation", "jtest_eval.py"), encoding="utf-8").read().split("\n")
M = os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py")
text = open(M, encoding="utf-8").read()
b_prov = src[[i for i, ln in enumerate(src) if ln.startswith("    prov = provenance(run, device)") and "STARTS" in ln][0]]
e_prov = src[[i for i, ln in enumerate(src) if ln.startswith("    prov = provenance(run, device)") and "as in dump_b" in ln][0]]
rows = [
    ("dump_b check_writable removed", "    check_writable(out)\n    refuse_existing(out)\n" + b_prov, "    refuse_existing(out)\n" + b_prov),
    ("dump_ends check_writable removed", "    check_writable(out)\n    refuse_existing(out)\n" + e_prov, "    refuse_existing(out)\n" + e_prov),
    ("expect same card message", ' (the card\'s file hashes equal the current ones: that digest is not this card\'s)"', '"'),
    ("atomic check_writable removed", "    check_writable(path)\n    if not overwrite and os.path.exists(path):", "    if not overwrite and os.path.exists(path):"),
]
anchor = "# glue\n"
add = "".join(f" ({n!r}, {o!r}, {w!r}),\n".replace("'" + n + "'", '"' + n + '"', 1) for n, o, w in rows)
assert text.count(anchor) == 1
open(M, "w", encoding="utf-8").write(text.replace(anchor, add + anchor))
print("ok")
