def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/cli.py", [
    ('''def main(argv=None) -> int:
    a = build_parser().parse_args(argv)''', '''def main(argv=None) -> int:
    from .apicli import dispatch                                       # `Evaluation.run ...`, `methods`: the class interface (api.py)
    code = dispatch(sys.argv[1:] if argv is None else list(argv))
    if code is not None:
        return code
    a = build_parser().parse_args(argv)'''),
    ('''    python -m evalsuite --adapter pkg.module:build [--arg key=value ...] [--out DIR] [--state carry|reset|warmup] <command>
''', '''    python -m evalsuite Evaluation.<method> ... | methods                 the class interface of a project (api.py, apicli.py): run, check, quant, compare, ...
    python -m evalsuite --adapter pkg.module:build [--arg key=value ...] [--out DIR] [--state carry|reset|warmup] <command>
'''),
])
print("ok")
