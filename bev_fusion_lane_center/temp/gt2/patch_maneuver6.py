p = "temp/jr/maneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('VARIANTS = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle", "oracle+fit"]', 'ALL = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle", "oracle+fit"]\nVARIANTS = [v for v in ALL if len(sys.argv) <= 3 or v in sys.argv[3:]]         # optional: the variants to compute (default all)')
sub('''prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (scales_or[(int(i), int(j))] if name == "oracle+fit" else ()) for i, j in zip(ia, ib)]''',
    '''prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (scales_or[(int(i), int(j))] if name == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and name.startswith("p")) else ())) for i, j in zip(ia, ib)]''')
sub('    for name, lg, thr in (("p>=0.5"', '    for name, lg, thr in [x for x in (("p>=0.5"')
sub('("oracle+fit", logit_or, 0.5)):', '("oracle+fit", logit_or, 0.5)) if x[0] in VARIANTS]:')
sub('    out = {"plain": QGE.to_raw(graph)}', '    out = {"plain": QGE.to_raw(graph)}')
sub('"oracle+fit": "#d62728"}', '"oracle+fit": "#d62728"}\ncols = {k_: v_ for k_, v_ in cols.items() if k_ in VARIANTS}')
open(p, "w", encoding="utf-8").write(s)
print("ok")
