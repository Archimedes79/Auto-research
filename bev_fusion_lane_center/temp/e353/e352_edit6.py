p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def runs_of(seed):
    return f"e324_s_s{seed}", f"e273_p_s{seed}"
''', '''def runs_of(seed):
    return f"e324_s_s{seed}", f"e273_p_s{seed}"


def point_run(seed, name, groups=False):
    """the folder name under `runs/` of a point: e352_<point>_s<S> (single channels, E345's ranking) or e352g_<point>_s<S> (whole GroupNorm groups)"""
    return f"e352{'g' if groups else ''}_{name}_s{seed}"
''')
s = s.replace('''def score_lines(seed, name):''', '''def score_lines(seed, name, groups=False):''')
s = s.replace('''    run = f"e352_{name}_s{seed}"
    head = f"bash''', '''    run = point_run(seed, name, groups)
    head = f"bash''')
s = s.replace("--tag {name}\"", "--tag {name}\"")
s = s.replace('''folder = PP.rootpath(a.out) if a.out else os.path.join(ROOT, "runs", f"e352_{a.point}_s{a.seed}")''', '''folder = PP.rootpath(a.out) if a.out else os.path.join(ROOT, "runs", point_run(a.seed, a.point, a.groups))''')
s = s.replace('''    s.add_argument("--point", default=None, choices=sorted(POINTS))
    sp.add_parser("selftest")''', '''    s.add_argument("--point", default=None, choices=sorted(POINTS))
    s.add_argument("--groups", action="store_true")
    sp.add_parser("selftest")''')
s = s.replace("for k, line in score_lines(a.seed, name).items():", "for k, line in score_lines(a.seed, name, a.groups).items():")
s = s.replace("--device cuda] [--groups] [--workers N] [--max N --out DIR]    # one point of one seed -> runs/e352_t1_s0/", "--device cuda] [--groups] [--workers N] [--max N --out DIR]    # one point of one seed -> runs/e352_t1_s0/ (--groups: runs/e352g_t1_s0/)")
open(p, "w", encoding="utf-8").write(s)
