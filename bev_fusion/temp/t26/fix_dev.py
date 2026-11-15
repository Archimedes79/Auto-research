p = "temp/t26/make_dev.py"
s = open(p).read()
old = 's = s.replace("    for group in (data_checks, marker_checks, cli_checks):\n        for name, v in group().items():\n            check(v, name)\n", "    pass\n")'
assert old in s, "pattern"
new = 's = s.replace("    for group in (data_checks, marker_checks, crash_checks, lock_checks, cli_checks):", "    for group in (marker_checks, crash_checks, lock_checks, cli_checks):")'
s = s.replace(old, new)
s = s.replace('for fn in ("jitter_fn_checks", "jitter_model_checks", "clip_checks"):', 'for fn in ("jitter_fn_checks", "jitter_model_checks", "clip_checks", "train_jitter_checks"):')
open(p, "w").write(s)
