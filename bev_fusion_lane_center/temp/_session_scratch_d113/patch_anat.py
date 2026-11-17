import io
p = "temp/_session_scratch_d113/anatomy_long.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('pvals = {k: [] for k in cnt}', 'pvals = {k: [] for k in cnt}\ntyp = {k: [] for k in cnt}')
s = s.replace('        pvals[k].append(p_)', '        pvals[k].append(p_)\n        pt = np.asarray(f["ptype"], float)\n        typ[k].append((pt[s_, 4], pt[d_, 5], pt[s_, 4] * pt[d_, 5]))')
s += '''
print("typed factor p_entry(src) * p_exit(dst) of the head's own type softmax: median / share above 0.1 / above 0.3 per category")
for k in ("tp", "dup_of_true", "wrong_pair", "false_end"):
    a = np.array(typ[k]) if typ[k] else np.zeros((1, 3))
    print("  %-12s p_entry %.2f p_exit %.2f product median %.3f | > 0.1: %.3f | > 0.3: %.3f" % (k, np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2]), (a[:, 2] > 0.1).mean(), (a[:, 2] > 0.3).mean()))
'''
io.open(p, "w", encoding="utf-8").write(s)
