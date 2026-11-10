def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/apicli.py", [
    ("`Evaluation.quant t23_C_s1@e3f9a2c1d4b0 --match drivable`, `Evaluation.compare CONTROL VARIANT`;", "`Evaluation.quant t23_C_s1@e3f9a2c1d4b0 --match drivable`, `Evaluation.compare A B` (b - a);"),
    ("A parameter without a default (and `network`, `id`, `results`, `frame`, `frames`,\n`control`, `variant`) is positional,", "A parameter without a default (and `network`, `id`, `results`, `a`, `b`, `frame`,\n`frames`) is positional,"),
    ('''POSITIONAL = ("network", "id", "results", "control", "variant", "frame", "frames")''', '''POSITIONAL = ("network", "id", "results", "a", "b", "frame", "frames")'''),
    ('''    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()''', '''    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):                 # numpy arrays and scalars
        return _clean(obj.tolist())'''),
    ('''        elif isinstance(p.default, bool):
            ap.add_argument(flag, dest=k, action="store_true", default=False)
        elif isinstance(p.default, (tuple, list)):''', '''        elif isinstance(p.default, (tuple, list)):'''),
    ('''    try:
        with contextlib.redirect_stdout(sys.stderr):                                     # whatever the engine or the project prints is not the result
            result = getattr(Evaluation(a.project, a.adapter), name)(**kw)
    except (EvaluationError, ValueError, KeyError, OSError, SystemExit, ImportError) as e:
        if os.environ.get("EVALSUITE_TRACEBACK"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args and isinstance(e.args[0], str) else str(e)
        print(f"error: {msg}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(_clean(result), indent=1, allow_nan=False))
    else:
        _text(_clean(result))
    return 1 if name == "check" and not result.get("ok") else 0''', '''    try:
        with contextlib.redirect_stdout(sys.stderr):                                     # whatever the engine or the project prints is not the result
            result = _clean(getattr(Evaluation(a.project, a.adapter), name)(**kw))
    except (Exception, SystemExit) as e:                                                 # an error of the data, the network or the machine is a message and exit 2, never a traceback
        if os.environ.get("EVALSUITE_TRACEBACK"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args and isinstance(e.args[0], str) else str(e.code if isinstance(e, SystemExit) and e.code is not None else e)
        print(f"error: {type(e).__name__ + ': ' if not isinstance(e, EvaluationError) else ''}{msg}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(result, indent=1, allow_nan=False))
    else:
        _text(result)
    return 1 if name == "check" and not result.get("ok") else 0'''),
])
print("ok")
