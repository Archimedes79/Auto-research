def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)

edit("training/grid/jtoken.py", [(
'''    per = (x * m).sum(1) / (m.sum(1) if w is None else (w * m).sum(1)).clamp_min(1)''',
'''    per = (x * m).sum(1) / (m.sum(1).clamp_min(1) if w is None else (w * m).sum(1).clamp_min(1e-12))''')])

edit("training/train/jtoken_train.py", [
('''    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    return ap.parse_args(argv)
''', '''    ap.add_argument("--smoke", action="store_true", help="20 steps on synthetic batches")
    a = ap.parse_args(argv)
    check_options(vars(a))
    return a


def check_options(args):
    """the options of the head and the loss, before anything is built or loaded: d a multiple of the heads, a positive ffn (0 is not 'the default': leave --ffn out), layers >= 1, a token drop in [0, 1)"""
    assert args["heads"] > 0 and args["d"] > 0 and args["d"] % args["heads"] == 0, f"--d {args['d']} is not a multiple of --heads {args['heads']}"
    assert args["layers"] >= 1, f"--layers {args['layers']}"
    assert args["ffn"] is None or args["ffn"] > 0, f"--ffn {args['ffn']}: a positive width, or leave it out for 2 x d"
    assert 0.0 <= args["token_drop"] < 1.0, f"--token-drop {args['token_drop']}: a probability in [0, 1)"
'''),
('''        old = {k: (c.get("args") or {}).get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}                    # a run of before the architecture options has the defaults''',
 '''        old = {k: (c.get("args") or {}).get(k, ARCH_DEFAULTS.get(k)) for k in RECIPE}                    # a run of before the architecture options has the defaults; conservative: ffn None and the explicit 2 x d are the same head but count as another recipe'''),
('''    assert d % g("heads") == 0, f"--d {d} is not a multiple of --heads {g('heads')}"
    return JK.JTokenHead(d=d, heads=g("heads"), layers=g("layers"), ffn=args.get("ffn") or 2 * d)''',
 '''    assert d % g("heads") == 0, f"--d {d} is not a multiple of --heads {g('heads')}"
    ffn = 2 * d if args.get("ffn") is None else args["ffn"]
    assert ffn > 0, f"ffn {ffn}"
    return JK.JTokenHead(d=d, heads=g("heads"), layers=g("layers"), ffn=ffn)'''),
])
