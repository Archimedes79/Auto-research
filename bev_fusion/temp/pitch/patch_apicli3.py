def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    ('''RESULT_DROP = ("gt_signature",''', '''RESULT_DROP = ("class", "gt_signature",'''),
])
patch("evalsuite/apicli.py", [
    ('''"""The command line of the interface (api.py): every method of `Evaluation` is a command with the same name. `python -m evalsuite Evaluation.run runs/t23_C_s1 small.val`,''',
     '''"""The command line of the interface (api.py, runner_api.py): every method of `Evaluation` and of `Runner` is a command with the same name. `python -m evalsuite Evaluation.run runs/t23_C_s1 small.val`,
`Runner.run runs/t23_C_s1 small.val --taps fused` (predictions without a ground truth),'''),
    ('''A method that no older command of this tool uses also works
without the class name (`run`, `check`, `by`, ...; `list`, `quant`, `probe`, `selftest` need `Evaluation.`).''', '''A method that no older command of this tool uses also works
without the class name (`run`, `check`, `by`, `tap`, ...: the Evaluation's when both classes have it; `list`, `quant`, `probe`, `selftest` need the class name).'''),
    ('''from .api import Evaluation, EvaluationError
''', '''from .api import Evaluation, EvaluationError
from .runner_api import Runner

CLASSES = {"Evaluation": Evaluation, "Runner": Runner}
'''),
    ('''def _signature(name: str) -> str:
    return ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
                     for k, p in inspect.signature(getattr(Evaluation, name)).parameters.items() if k != "self")


def methods_text() -> None:
    for name in Evaluation.METHODS:
        print(f"Evaluation.{name}({_signature(name)})")
        print("    " + (inspect.getdoc(getattr(Evaluation, name)) or "").splitlines()[0])


def _argparser(name: str) -> argparse.ArgumentParser:
    fn = getattr(Evaluation, name)
    ap = argparse.ArgumentParser(prog=f"python -m evalsuite Evaluation.{name}", description=inspect.getdoc(fn), formatter_class=argparse.RawDescriptionHelpFormatter)''', '''def _signature(cls, name: str) -> str:
    return ", ".join(p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
                     for k, p in inspect.signature(getattr(cls, name)).parameters.items() if k != "self")


def methods_text() -> None:
    for cname, cls in CLASSES.items():
        for name in cls.METHODS:
            print(f"{cname}.{name}({_signature(cls, name)})")
            print("    " + (inspect.getdoc(getattr(cls, name)) or "").splitlines()[0])


def _argparser(cls, name: str) -> argparse.ArgumentParser:
    fn = getattr(cls, name)
    ap = argparse.ArgumentParser(prog=f"python -m evalsuite {cls.__name__}.{name}", description=inspect.getdoc(fn), formatter_class=argparse.RawDescriptionHelpFormatter)'''),
    ('''    head = argv[0]
    prefixed = head.startswith("Evaluation.")
    name = (head.split(".", 1)[-1] if prefixed else head).replace("-", "_")
    if name not in Evaluation.METHODS or (not prefixed and head in LEGACY):
        return None
    a = _argparser(name).parse_args(argv[1:])
    kw = {}
    for k, p in inspect.signature(getattr(Evaluation, name)).parameters.items():''', '''    head = argv[0]
    cname, dot, rest = head.partition(".")
    if dot and cname in CLASSES:
        cls, name = CLASSES[cname], rest.replace("-", "_")
        if name not in cls.METHODS:
            return None
    else:                                                                              # a bare word: the older commands keep theirs; else the Evaluation's method, else the Runner's
        name = head.replace("-", "_")
        cls = next((c for c in CLASSES.values() if name in c.METHODS), None)
        if cls is None or head in LEGACY:
            return None
    a = _argparser(cls, name).parse_args(argv[1:])
    kw = {}
    for k, p in inspect.signature(getattr(cls, name)).parameters.items():'''),
    ('''            result = _clean(getattr(Evaluation(a.project, a.adapter), name)(**kw))''', '''            result = _clean(getattr(cls(a.project, a.adapter), name)(**kw))'''),
])
print("ok")
