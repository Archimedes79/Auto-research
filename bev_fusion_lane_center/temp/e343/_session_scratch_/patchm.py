s = open("mutate.py", encoding="utf-8").read()
old = ''' ("cmd_all use_counts any", 'use_counts = all(d["control"]', 'use_counts = any(d["control"]'),'''
new = ''' ("counts_reproduced any", 'return all(d["control"] and d["control"]["counts"] is not None for d in done)', 'return any(d["control"] and d["control"]["counts"] is not None for d in done)'),
 ("cmd_all use_counts always", 'use_counts = bool(setup["arms"]) and counts_reproduced(done)', 'use_counts = bool(setup["arms"])'),
 ("seed_inputs ctl_b ignores flag", 'if cd["b"] and not control_ends_flag else i["b"]', 'if cd["b"] else i["b"]'),
 ("seed_inputs test ctl ends", 'ctl_ends = sub_seed(a.control_ends_dump, seed) or (os.path.join(ROOT, sub_seed(cd["ends"], seed)) if cd["ends"] else None)', 'ctl_ends = sub_seed(a.control_ends_dump, seed) or os.path.join(ROOT, sub_seed(CONTROL_DEFAULTS["val"]["ends"], seed))'),
 ("seed_inputs smoke folder", 'd, run = dump_dir(a.split, a.max),', 'd, run = dump_dir(a.split),'),'''
assert old in s
open("mutate.py", "w", encoding="utf-8").write(s.replace(old, new))
