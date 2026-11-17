s = open("mutate.py", encoding="utf-8").read()
old = ''' ("provenance missing sidecar ok",'''
new = ''' ("get_dump flagged unchecked", 'elif expect is not None and (make is not None or os.path.exists(head_path(path))):', 'elif expect is not None and make is not None:'),
 ("get_dump default no sidecar ok", 'elif expect is not None and (make is not None or os.path.exists(head_path(path))):', 'elif expect is not None and os.path.exists(head_path(path)):'),
 ("score_seed B flag unchecked", '"B\'s dump", n, exp(run_b))}', '"B\'s dump", n, exp(run_b) if i["b_made"] else None)}'),
 ("score_seed ctl ends unchecked", 'get_dump(i["ctl_ends"], None, split, "the control\'s pair dump", n, exp(sub_seed(CONTROL_RUN, seed)))', 'load_dump(i["ctl_ends"], split, "the control\'s pair dump", n)'),
 ("score_seed ctl B unchecked", 'get_dump(i["ctl_b"], None, split, "the control\'s B dump", n, exp(run_b))', 'load_dump(i["ctl_b"], split, "the control\'s B dump", n)'),
 ("provenance missing sidecar ok",'''
assert s.count(old) == 1
open("mutate.py", "w", encoding="utf-8").write(s.replace(old, new))
