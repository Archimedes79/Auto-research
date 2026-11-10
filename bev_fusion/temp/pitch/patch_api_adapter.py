def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    ('''def load_adapter(spec: Optional[str], project: Optional[str]):
    project = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    if spec is None:''', '''ADAPTER_FUNCTIONS = ("name", "out", "network", "sets", "source", "subject", "task", "identity")


def load_adapter(spec: Any, project: Optional[str]):
    """the adapter module: given (a module or object with the functions), named in `spec` (a module path), or named in <project>/evaluation/evalsuite.json"""
    project = os.path.abspath(project or os.environ.get("AR_PROJECT") or ".")
    if spec is not None and not isinstance(spec, str):
        missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(spec, f)]
        if missing:
            raise EvaluationError(f"the adapter lacks {', '.join(missing)}")
        return spec
    if spec is None:'''),
    ('''    missing = [f for f in ("name", "out", "network", "sets", "source", "subject", "task", "identity") if not hasattr(mod, f)]''',
     '''    missing = [f for f in ADAPTER_FUNCTIONS if not hasattr(mod, f)]'''),
])
print("ok")
