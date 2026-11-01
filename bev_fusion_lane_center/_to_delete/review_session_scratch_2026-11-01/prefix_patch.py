p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
i, j = s.index("def repo_modules("), s.index("def bundle(")
new = '''def repo_modules(dotted_names, repo, ref=None, prefixes=None):
    """the files of the repository that absolute imports like `autogt.layers` name: for each dotted name the longest prefix that is a module file or a
    package, plus the __init__.py of its parent packages; paths relative to the repository root, sorted. The name is looked for under the repository
    root and under the problem folder (code puts it on sys.path: bev_fusion/autogt before autogt moved to the top level). Names that are not
    repository code (torch.nn, numpy) find nothing. With `ref` (a tag) the files are looked for in that tag, not in the working tree"""
    out = set()
    if prefixes is None:
        pr = os.path.relpath(P.root(), repo).replace("\\\\", "/")
        prefixes = [""] + ([] if pr in (".", "") or pr.startswith("..") else [pr + "/"])

    def has(rel):
        return rel in tree_names(repo, ref) if ref else os.path.isfile(os.path.join(repo, rel))
    for dotted in dotted_names:
        parts = dotted.split(".")
        for pre in prefixes:
            found = False
            for i in range(len(parts), 0, -1):
                rels = [pre + "/".join(parts[:i]) + ".py", pre + "/".join(parts[:i]) + "/__init__.py"]
                f = next((r for r in rels if has(r)), None)
                if f:
                    out.add(f)
                    for k in range(1, i):
                        init = pre + "/".join(parts[:k]) + "/__init__.py"
                        if has(init):
                            out.add(init)
                    found = True
                    break
            if found:
                break
    return sorted(out)


'''
s = s[:i] + new + s[j:]
# the closure and the unfollowed check use the prefixes through the default; local-name detection accepts a match under either root
old = 'not any(f.split("/")[0] == t for f in repo_files))'
assert s.count(old) == 1
s = s.replace(old, 'not any(t in f.split("/")[:2] for f in repo_files))')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
