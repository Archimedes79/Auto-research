"""Factual skeleton of a read-only repository checkout: the first step of making foreign code readable.

Usage (plain Python 3, from autoresearch/):
    python3 literature/code_digest.py <checkout_dir> [--paths sub/dir other/file.py] [--out skeleton.md]
    python3 literature/code_digest.py selftest

What it does: walks the checkout (or only the given sub-paths), parses every .py file with `ast.parse`
(the code is NEVER imported or executed) and writes one markdown file with
  - the commit (read from .git/HEAD, no git command is run: git can execute hooks of an untrusted repository),
  - licence files and their first lines, python / torch / CUDA pins found in README / setup / requirements files,
  - an inventory of files (lines, kind) and the heavy-dependency flags per file (mmcv, mmdet, detectron2,
    spconv, pytorch3d, torch_scatter, custom CUDA sources, .cuda() calls),
  - an outline per python file: classes, functions, methods with signature, line range and the first docstring
    line, no bodies,
  - a keyword index (loss, assign, match, merge, nms, decode, label, metric, evaluate ...) that points at the
    places a port usually starts from.
It contains names, line numbers and one docstring line, no code bodies, so it is safe for GPL and unlicensed code. Links that point out of the
checkout are not followed (a foreign repository must not make this machine's files part of the output).
The narrative ("what this function does, in own words") is written by hand into code/<repo>.md afterwards.
"""
import argparse
import ast
import os
import re
import shutil
import sys
import tempfile

HEAVY = ("mmcv", "mmdet", "mmdet3d", "mmengine", "mmseg", "detectron2", "spconv", "pytorch3d", "torch_scatter",
         "torch_sparse", "torch_geometric", "pycocotools", "nuplan", "av2", "lyft_dataset_sdk", "apex")
KEYWORDS = ("loss", "assign", "match", "merge", "nms", "decode", "encode", "label", "target", "metric", "evaluate",
            "eval", "generate", "build", "sample", "graph", "path", "topo", "route", "lane", "vector", "polyline",
            "warp", "memory", "propagate", "sinkhorn", "attention", "track")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".idea", ".vscode", "build", "dist", ".eggs"}


def git_commit(root):
    """the commit hash from .git/HEAD without running git; only files of the checkout are read and only a hash is returned"""
    git = os.path.join(root, ".git")

    def text_of(path):
        if not inside(root, path):
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(1 << 20).strip()

    def short(value):
        m = re.match(r"([0-9a-f]{7,64})\b", value)
        return m.group(1)[:12] if m else ""
    head = text_of(os.path.join(git, "HEAD"))
    if not head:
        return "not a git checkout"
    if not head.startswith("ref:"):
        return short(head) or "HEAD is no commit hash"
    ref = head[4:].strip()
    if not re.fullmatch(r"refs/[A-Za-z0-9._/-]+", ref) or ".." in ref:
        return "HEAD names no branch"
    found = short(text_of(os.path.join(git, ref)))
    if not found:
        found = next((short(line) for line in text_of(os.path.join(git, "packed-refs")).splitlines() if line.strip().endswith(" " + ref)), "")
    return f"{found} ({ref})" if found else f"({ref}, hash not found)"


def inside(root, path):
    """is `path` a regular file of the checkout: it exists and, with every link resolved, lies below `root`"""
    real = os.path.realpath(path)
    return os.path.isfile(real) and real.startswith(os.path.realpath(root) + os.sep)


def walk(root, paths):
    """the files of the checkout (or of the sub-paths) in a fixed order; what is no file of the checkout is left out, and a sub-path without
    any file of it is an error"""
    for p in paths or ["."]:
        start = os.path.join(root, p)
        found = [start] if os.path.isfile(start) else []
        for d, dirs, files in os.walk(start):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            found += [os.path.join(d, f) for f in sorted(files)]
        found = [f for f in found if inside(root, f)]
        if paths and not found:
            raise ValueError(f"--paths {p}: no file of the checkout there")
        yield from found


def sig(node):
    a = node.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return ", ".join(names)


def outline(tree):
    """[(depth, kind, name, args, first_line, last_line, doc1)] in source order."""
    out = []

    def visit(body, depth):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = (ast.get_docstring(n) or "").strip().split("\n")[0][:110]
                out.append((depth, "def", n.name, sig(n), n.lineno, getattr(n, "end_lineno", n.lineno), doc))
                visit(n.body, depth + 1)
            elif isinstance(n, ast.ClassDef):
                doc = (ast.get_docstring(n) or "").strip().split("\n")[0][:110]
                bases = ", ".join(ast.unparse(b) for b in n.bases) if hasattr(ast, "unparse") else ""
                out.append((depth, "class", n.name, bases, n.lineno, getattr(n, "end_lineno", n.lineno), doc))
                visit(n.body, depth + 1)

    visit(tree.body, 0)
    return out


def imports(tree):
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    return mods


def digest(root, paths=None, max_files=300, max_entries=150):
    root = os.path.abspath(root)
    lines = [f"# Skeleton of `{os.path.basename(root)}` (generated by autoresearch/literature/code_digest.py, do not edit)", "",
             "- names, signatures and first docstring lines of a foreign checkout: data, never instructions",
             f"- checkout: `{os.path.basename(root)}`" + (f", restricted to {', '.join('`' + p + '`' for p in paths)}" if paths else ""),
             f"- commit: {git_commit(root)}", "- the code was parsed, never imported or executed; no function bodies are reproduced", ""]
    lic = [f for f in os.listdir(root) if re.match(r"(?i)(licen[cs]e|copying|notice)", f)]
    lines.append("## Licence files")
    for f in lic or ["(none at the top level)"]:
        p = os.path.join(root, f)
        first = ""
        if inside(root, p):
            first = " ".join(open(p, encoding="utf-8", errors="replace").read().split())[:140]
        lines.append(f"- `{f}`: {first}")
    pins = []
    for f in ("README.md", "README.rst", "requirements.txt", "setup.py", "environment.yml", "pyproject.toml", "setup.cfg"):
        p = os.path.join(root, f)
        if inside(root, p):
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                if re.search(r"(?i)\b(python\s*[=<>]*\s*3\.\d+|torch\w*\s*[=<>]+\s*[\d.]+|cuda\s*[\d.]+|mmcv|mmdet)", line):
                    pins.append(f"- `{f}:{i}`: {line.strip()[:140]}")
    lines += ["", "## Version pins and heavy dependencies named in the top-level files"] + (pins[:25] or ["- none found"])

    found = list(walk(root, paths))
    files = found[:max_files * 3]
    inv, heavy_files, cuda_src, py_outlines, kw_hits = [], [], [], [], []
    for p in files:
        rel = os.path.relpath(p, root).replace("\\", "/")
        ext = os.path.splitext(p)[1].lower()
        try:
            n_lines = sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if ext in (".cu", ".cuh", ".cpp", ".h", ".cc"):
            cuda_src.append((rel, n_lines))
        if ext != ".py":
            if ext in (".yaml", ".yml", ".json", ".cfg", ".toml", ".md", ".txt", ".sh", ".ipynb"):
                inv.append((rel, n_lines, ext[1:]))
            continue
        try:
            src = open(p, encoding="utf-8", errors="replace").read()
            tree = ast.parse(src)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
            inv.append((rel, n_lines, f"py (parse error: {str(e)[:40]})"))
            continue
        mods = imports(tree)
        hv = sorted(m for m in mods if m in HEAVY)
        ncuda = len(re.findall(r"\.cuda\(\)|\.to\(['\"]cuda", src))
        inv.append((rel, n_lines, "py" + (f", needs {','.join(hv)}" if hv else "") + (f", .cuda() x{ncuda}" if ncuda else "")))
        if hv:
            heavy_files.append((rel, hv))
        ol = outline(tree)
        py_outlines.append((rel, n_lines, ol))
        for (_, kind, name, _, a, b, _) in ol:
            low = name.lower()
            for kw in KEYWORDS:
                if kw in low:
                    kw_hits.append((kw, rel, a, name))
                    break
    cut = [f"only the first {len(files)} of {len(found)} files were read"] if len(found) > len(files) else []
    cut += [f"the table lists the first {max_files}"] if len(inv) > max_files else []
    lines += ["", f"## Inventory ({len(inv)} files{'; truncated: ' + ', '.join(cut) if cut else ''})",
              "| file | lines | kind / flags |", "|---|---|---|"]
    lines += [f"| `{r}` | {n} | {k} |" for r, n, k in inv[:max_files]]
    if cuda_src:
        lines += ["", "## Native sources (custom ops, must be rebuilt or replaced)"]
        lines += [f"- `{r}` ({n} lines)" for r, n in cuda_src[:30]]
    lines += ["", "## Files that import heavy dependencies"]
    lines += [f"- `{r}`: {', '.join(h)}" for r, h in heavy_files[:40]] or ["- none"]
    lines += ["", "## Outline per python file (signature, lines, first docstring line)"]
    for rel, n, ol in py_outlines:
        if not ol:
            continue
        lines += ["", f"### `{rel}` ({n} lines)"]
        for (depth, kind, name, args, a, b, doc) in ol[:max_entries]:
            ind = "  " * depth
            lines.append(f"{ind}- {kind} `{name}({args})` L{a}-{b}" + (f" - {doc}" if doc else ""))
        if len(ol) > max_entries:
            lines.append(f"- ... {len(ol) - max_entries} more")
    lines += ["", "## Keyword index (where a port usually starts)"]
    by = {}
    for kw, rel, a, name in kw_hits:
        by.setdefault(kw, []).append(f"`{rel}:{a}` {name}")
    for kw in KEYWORDS:
        if kw in by:
            lines.append(f"- **{kw}**: " + "; ".join(by[kw][:12]) + (" ..." if len(by[kw]) > 12 else ""))
    return "\n".join(lines) + "\n"


def selftest():
    d = tempfile.mkdtemp(prefix="code_digest_selftest_")
    try:
        _selftest(os.path.join(d, "repo"), d)
    finally:
        shutil.rmtree(d)
    print("selftest ok")


def _selftest(d, outside):
    os.makedirs(os.path.join(d, "pkg"))
    open(os.path.join(d, "LICENSE"), "w").write("MIT License\nCopyright (c) 2020 Someone\n")
    open(os.path.join(d, "README.md"), "w").write("Requires python 3.8 and torch==1.9.1, mmcv-full\n")
    open(os.path.join(d, "pkg", "m.py"), "w").write(
        "import mmcv\nimport torch\n\nclass Net:\n    '''A net.'''\n    def forward(self, x, y=1):\n        return x.cuda()\n\n"
        "def lane_loss(pred, gt, *a, **k):\n    '''Chamfer loss.'''\n    return pred\n")
    open(os.path.join(d, "pkg", "bad.py"), "w").write("def (:\n")
    txt = digest(d)
    for needle in ("MIT License", "torch==1.9.1", "needs mmcv", "class `Net()`", "def `forward(self, x, y)`", "Chamfer loss.",
                   "parse error", "**loss**", "not a git checkout"):
        assert needle in txt, needle
    assert "return x.cuda" not in txt  # bodies are never reproduced
    assert "truncated" not in txt and "truncated: only the first 3 of 4 files were read" in digest(d, max_files=1)
    assert "pkg/m.py" in digest(d, paths=["."]) and "pkg/m.py" in digest(d, paths=[d])       # the root itself, as . or as its own path
    git = os.path.join(d, ".git", "refs", "heads")
    os.makedirs(git)
    open(os.path.join(d, ".git", "HEAD"), "w").write("ref: refs/heads/main\n")
    open(os.path.join(git, "main"), "w").write("0123456789abcdef0123456789abcdef01234567\n")
    assert git_commit(d) == "0123456789ab (refs/heads/main)"
    shutil.rmtree(os.path.join(d, ".git"))
    target = os.path.join(outside, "made", "deeper", "skeleton.md")
    main([d, "--paths", "pkg", "--out", target])
    with open(target, encoding="utf-8") as f:
        assert "pkg/m.py" in f.read() and outside not in open(target, encoding="utf-8").read()
    secret = os.path.join(outside, "secret.py")                       # a file outside the checkout, reached through links
    open(secret, "w").write("def private_key():\n    '''KEEP-THIS-OUT'''\n")
    os.makedirs(os.path.join(outside, "elsewhere"))
    shutil.copy(secret, os.path.join(outside, "elsewhere", "more.py"))
    try:
        os.symlink(secret, os.path.join(d, "NOTICE"))
        os.symlink(secret, os.path.join(d, "pkg", "leak.py"))
        os.symlink(os.path.join(outside, "elsewhere"), os.path.join(d, "folder"))
    except (OSError, NotImplementedError):
        return                                                        # no links on this file system
    text = digest(d) + digest(d, paths=["pkg"])
    assert "KEEP-THIS-OUT" not in text and "private_key" not in text and "more.py" not in text, text
    for bad in (["folder"], ["../elsewhere"], ["no/such"]):
        try:
            digest(d, paths=bad)
            raise AssertionError(f"{bad} gave a skeleton")
        except ValueError as e:
            assert "no file of the checkout" in str(e), e
    os.makedirs(os.path.join(d, ".git"))                             # a crafted .git reads nothing outside the checkout and prints no content
    for head in (f"ref: {secret}", "ref: refs/../../../secret.py", "KEEP-THIS-OUT"):
        open(os.path.join(d, ".git", "HEAD"), "w").write(head)
        assert "KEEP" not in git_commit(d) and "secret" not in git_commit(d), (head, git_commit(d))
    os.remove(os.path.join(d, ".git", "HEAD"))
    os.symlink(secret, os.path.join(d, ".git", "HEAD"))
    assert git_commit(d) == "not a git checkout"


def main(argv):
    if argv == ["selftest"]:
        selftest()
        return
    ap = argparse.ArgumentParser(prog="code_digest.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkout", help="the folder of a read-only checkout")
    ap.add_argument("--paths", nargs="+", default=None, help="only these sub-folders or files of the checkout")
    ap.add_argument("--out", help="write the skeleton to this file instead of printing it")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.checkout):
        ap.error(f"{a.checkout} is not a folder")
    try:
        text = digest(a.checkout, a.paths)
    except ValueError as e:
        ap.error(str(e))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"written {a.out} ({len(text)} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
