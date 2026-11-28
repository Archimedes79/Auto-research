"""What a textual mutant is made from, since D152 split a model module from its losses (net/loss/loss_<module>.py): the text of both as ONE module.
    from lane_center.net.tests.merged_source import merged
    src = merged(path_of_the_model_module)          # the text of the module alone when it has no loss module
The model's imports of its loss module are dropped (the names are defined in the merged text) and so are the loss module's imports of the model; the
losses go in front of the `__main__` block, so a self-test of the model runs them. `must_fail` is the mutation check the tests share."""
import ast
import os
import tempfile

# the package of the model and loss modules: they import each other as NET.model.<m> / NET.loss.loss_<m>
NET = "lane_center.net"


def must_fail(fn, *a, **k):
    """a mutation check: fn must fail its assertion"""
    try:
        fn(*a, **k)
    except AssertionError:
        return
    raise RuntimeError(f"{getattr(fn, '__name__', fn)} passed with the mutated input")


def merged(model_path):
    with open(model_path, encoding="utf-8") as fh:
        src = fh.read()
    name = os.path.basename(model_path)[:-3]
    loss_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), os.pardir, "loss", f"loss_{name}.py")
    if not os.path.exists(loss_path):
        return src
    with open(loss_path, encoding="utf-8") as fh:
        loss = fh.read()

    def without(text, module):
        """`text` without the lines of every `from module import ...` statement; `from <package> import <module>` (the other half bound to a
        name, which the merged text cannot give) is refused"""
        lines = text.split("\n")
        drop = set()
        pkg, _, leaf = module.rpartition(".")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module == pkg and any(a.name == leaf for a in node.names):
                raise AssertionError(f"`from {pkg} import {leaf}` in {name}: the merged text cannot bind it; import its names (`from {module} import ...`)")
            if isinstance(node, ast.ImportFrom) and node.module == module:
                drop |= set(range(node.lineno - 1, node.end_lineno))
        return "\n".join(l for i, l in enumerate(lines) if i not in drop)
    body = without(loss, f"{NET}.model.{name}")
    first = ast.parse(body).body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):                  # the loss module's docstring
        body = "\n".join(body.split("\n")[first.end_lineno:])
    src = without(src, f"{NET}.loss.loss_{name}")
    mark = '\nif __name__ == "__main__":'
    if mark in src:
        head, tail = src.split(mark, 1)
        return head + "\n" + body + "\n" + mark + tail
    return src + "\n\n" + body


def _selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(f[5:-3] for f in os.listdir(os.path.join(here, os.pardir, "loss")) if f.startswith("loss_")):
        path = os.path.join(here, os.pardir, "model", f"{name}.py")
        text = merged(path)
        tree = ast.parse(text)
        defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        assert len(defs) == len(set(defs)), (name, sorted({d for d in defs if defs.count(d) > 1}))
        imports = "\n".join(l for l in text.split("\n") if l.lstrip().startswith(("from ", "import ")))
        assert f"loss_{name}" not in imports and f"{NET}.model.{name} " not in imports, name                  # neither half imports the other
        compile(text, path, "exec")
    with tempfile.TemporaryDirectory() as tmp:                                                                     # the module form of the import is refused
        for folder, file, text in (("model", "x.py", "A = 1\n"), ("loss", "loss_x.py", f"from {NET}.model import x as X\nB = X.A\n")):
            os.makedirs(os.path.join(tmp, folder))
            with open(os.path.join(tmp, folder, file), "w", encoding="utf-8") as fh:
                fh.write(text)
        must_fail(merged, os.path.join(tmp, "model", "x.py"))
    print("merged_source selftest ok: every model with a loss module reads as one module, no definition twice; an import of the other half as a module is refused")


if __name__ == "__main__":
    _selftest()
