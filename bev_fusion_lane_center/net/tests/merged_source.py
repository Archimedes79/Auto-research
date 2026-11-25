"""What a textual mutant is made from, since D152 split a model module from its losses (net/loss/loss_<module>.py): the text of both as ONE module.
    from merged_source import merged
    src = merged(path_of_the_model_module)          # the text of the module alone when it has no loss module
The model's imports of its loss module are dropped (the names are defined in the merged text) and so are the loss module's imports of the model; the
losses go in front of the `__main__` block, so a self-test of the model runs them."""
import ast
import os


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
        """`text` without the lines of every `from module import ...` statement"""
        lines = text.split("\n")
        drop = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                drop |= set(range(node.lineno - 1, node.end_lineno))
        return "\n".join(l for i, l in enumerate(lines) if i not in drop)
    body = without(loss, name)
    first = ast.parse(body).body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):                  # the loss module's docstring
        body = "\n".join(body.split("\n")[first.end_lineno:])
    src = without(src, f"loss_{name}")
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
        assert f"loss_{name}" not in "\n".join(l for l in text.split("\n") if l.lstrip().startswith(("from ", "import "))), name
        compile(text, path, "exec")
    print("merged_source selftest ok: every model with a loss module reads as one module, no definition twice")


if __name__ == "__main__":
    _selftest()
