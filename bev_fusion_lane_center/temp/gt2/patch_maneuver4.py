p = "temp/jr/maneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('VARIANTS = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle"]', 'VARIANTS = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle", "oracle+fit"]')
sub("import qtargets as QT  # noqa: E402", "import qtargets as QT  # noqa: E402\nimport qjunction as QJ  # noqa: E402")
sub('''        for a, b in {(int(p[0]), int(p[1])) for p in jg["pairs"]}:
            if ma[a] >= 0 and mb[b] >= 0 and e["allowed"][ma[a], mb[b]]:
                logit_or[ma[a], mb[b]] = 20.0
''', '''        sc_all = QJ.fit_pair_scales(jg)                                                  # (npair, 2): the handle scales fitted to the GT connector of every true pair
        for ip, pr in enumerate(jg["pairs"]):
            a, b = int(pr[0]), int(pr[1])
            if ma[a] >= 0 and mb[b] >= 0 and e["allowed"][ma[a], mb[b]]:
                logit_or[ma[a], mb[b]] = 20.0
                scales_or[(int(ma[a]), int(mb[b]))] = tuple(float(x) for x in sc_all[ip])
''')
sub('    logit_or = np.full_like(e["logit"], -20.0)\n', '    logit_or = np.full_like(e["logit"], -20.0)\n    scales_or = {}\n')
sub('''    for name, lg, thr in (("p>=0.5", e["logit"], 0.5), ("p>=0.2", e["logit"], 0.2), ("p>=0.1", e["logit"], 0.1), ("oracle", logit_or, 0.5)):
        p = 1.0 / (1.0 + np.exp(-lg)) * e["allowed"]
        ia, ib = np.nonzero(p >= thr)
        prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) for i, j in zip(ia, ib)]''', '''    for name, lg, thr in (("p>=0.5", e["logit"], 0.5), ("p>=0.2", e["logit"], 0.2), ("p>=0.1", e["logit"], 0.1), ("oracle", logit_or, 0.5), ("oracle+fit", logit_or, 0.5)):
        p = 1.0 / (1.0 + np.exp(-lg)) * e["allowed"]
        ia, ib = np.nonzero(p >= thr)
        prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (scales_or[(int(i), int(j))] if name == "oracle+fit" else ()) for i, j in zip(ia, ib)]''')
sub('"oracle": "#ff7f0e"}', '"oracle": "#ff7f0e", "oracle+fit": "#d62728"}')
open(p, "w", encoding="utf-8").write(s)
print("ok")
