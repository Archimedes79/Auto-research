def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/evaluate.py", [
    ('''Global options as in bev_suite: --device cuda|cpu  --workers N  --split val|train  --ckpt-dir DIR  --out DIR. The TEST split is not offered.
Nothing asked for (no option): this help, exit 2.''',
     '''    --sets               the sets that exist (evaluation/bev_sets.py: the splits of splits/, the same definition the training reads); needs no run
Global options as in bev_suite: --device cuda|cpu  --workers N  --split SET  --ckpt-dir DIR  --out DIR. SET = val (the default) or train: the lists the run itself was trained
and validated on; or <split>.<part> of splits/: small.val, small.train, large.val, large.train, xl.val, ... (`--sets` lists them): the same run, the same checkpoint and
caches, other scenes. A named set writes its own files (eval/card_small_val.json, quant/eval_small_val_n80), so the run's plain evaluation is never touched; `--all` (the
chain of a finished run: generalisation, report page) and `--quant --curves ...` are for the run's own lists, a named set takes --quant / --qual / --videos / --time. The TEST
split is not offered.
Nothing asked for (no option): this help, exit 2.'''),
    ('''    p.add_argument("--split", default="val", choices=["val", "train"])''',
     '''    p.add_argument("--split", default="val", metavar="SET", help="val | train (the run's own lists) or <split>.<part> of splits/ (small.val, large.train, ...); --sets lists them")
    p.add_argument("--sets", action="store_true", help="list the sets and stop (no run needed)")'''),
    ('''    if a.all and a.split == "train":
        raise SystemExit("--all with --split train: generalisation already evaluates train; use --quant --split train")''',
     '''    if a.all and a.split != "val":
        raise SystemExit(f"--all is the chain of the run's own lists (val; generalisation evaluates train itself): with --split {a.split} use --quant / --qual / --videos")'''),
    ('''    ev, tag = os.path.join(a.run, "eval"), "_train" if a.split == "train" else ""''',
     '''    ev, tag = os.path.join(a.run, "eval"), bev_sets.tag(a.split)'''),
    ('''    named = {w: count_or_list(t, w) for w, t in (("qual", a.qual), ("videos", a.videos)) if t}''',
     '''    named = {w: count_or_list(t, w) for w, t in (("qual", a.qual), ("videos", a.videos)) if t}       # the lists below are those of the set --split names'''),
    ('''    if not os.path.isfile(os.path.join(a.run, "config.json")):''',
     '''    try:
        bev_sets.parse(a.split)
    except ValueError as ex:
        return str(ex)
    if not os.path.isfile(os.path.join(a.run, "config.json")):'''),
    ('''    p = build_parser()
    a = p.parse_args(argv)
    if not (a.run and (a.quant or a.qual or a.videos or a.time or a.all)):''',
     '''    p = build_parser()
    a = p.parse_args(argv)
    if a.sets:
        print(bev_sets.table())
        return 0
    if not (a.run and (a.quant or a.qual or a.videos or a.time or a.all)):'''),
    ('''SUITE = os.path.join(ROOT, "evaluation", "bev_suite.py")''',
     '''SUITE = os.path.join(ROOT, "evaluation", "bev_suite.py")
sys.path.insert(0, ROOT)
from evaluation import bev_sets  # noqa: E402                                      the one definition of the sets (the splits of the training)'''),
])
print("ok")
