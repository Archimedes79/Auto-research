import io
p = "temp/_session_scratch_d113/anatomy_long.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('typ = {k: [] for k in cnt}', 'typ = {k: [] for k in cnt}\nuu = {k: [] for k in cnt}')
s = s.replace('        pt = np.asarray(f["ptype"], float)', '        uo, ui = np.array(c.f["uout"][i], float), np.array(c.f["uin"][i], float)\n        uu[k].append((1 / (1 + np.exp(-uo[s_])), 1 / (1 + np.exp(-ui[d_]))))\n        pt = np.asarray(f["ptype"], float)')
s += '''
print("B' long-link evidence sigmoid(uout[src]) * sigmoid(uin[dst]): median per source / destination, share of the product above 0.1 / 0.3")
for k in ("tp", "dup_of_true", "wrong_pair", "false_end"):
    a = np.array(uu[k]) if uu[k] else np.zeros((1, 2))
    pr = a[:, 0] * a[:, 1]
    print("  %-12s uout %.2f uin %.2f | product > 0.1: %.3f | > 0.3: %.3f" % (k, np.median(a[:, 0]), np.median(a[:, 1]), (pr > 0.1).mean(), (pr > 0.3).mean()))
'''
io.open(p, "w", encoding="utf-8").write(s)
