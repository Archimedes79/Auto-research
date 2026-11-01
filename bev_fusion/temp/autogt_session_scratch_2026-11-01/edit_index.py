p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def build(dataroot''', '''def write_index(sig_dir, gen, version, names, stride, n_keyframes, options):
    """index.json of a cache folder (marks it as an autogt cache for `clean`, records what it holds)."""
    idx = {"signature": gen.signature(), "version": version, "res": gen.grid.res, "ss": gen.ss,
           "options": {k: repr(v) for k, v in options.items()}, "scenes": list(names), "stride": stride, "keyframes": n_keyframes}
    json.dump(idx, open(os.path.join(sig_dir, "index.json"), "w"), indent=1)


def build(dataroot''', 1)
a = s.index('    idx = {"signature": gen.signature(), "version": version, "res": res')
b = s.index('    if verbose:\n        print(f"{done} written')
s = s[:a] + '    write_index(sig_dir, gen, version, names, stride, len(tokens), options)\n' + s[b:]
open(p, "w", encoding="utf-8").write(s)
p = "autogt/report.py"
s = open(p, encoding="utf-8").read()
s = s.replace("from .cli import _flat  # noqa: E402", "from .cli import _flat, write_index  # noqa: E402")
s = s.replace("    fov = radar_fov(gen, names, stride)\n", "    if sig_dir:\n        write_index(sig_dir, gen, version, names, stride, n_kf, options)\n    fov = radar_fov(gen, names, stride)\n")
open(p, "w", encoding="utf-8").write(s)
print("ok")
