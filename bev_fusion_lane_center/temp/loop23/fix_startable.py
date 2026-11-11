p = "training/startable.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep('''def selftest():
    good = dict(''', '''def resolve(path):
    """a config path as given, else relative to the problem folder (`_py14.sh` runs a script in its own folder, `trainsuite start` names the config relative to the problem folder)"""
    return path if os.path.isfile(path) else os.path.join(ROOT, path)


def selftest():
    import tempfile
    tmp0 = tempfile.mkdtemp()                                   # every check below runs on an empty problem folder: a real run of the same name must not change the result
    good = dict(''')
s = s.replace("p, _ = check(good, inputs=[])", "p, _ = check(good, root=tmp0, inputs=[])")
s = s.replace("p, _ = check(dict(good, **bad), inputs=[])", "p, _ = check(dict(good, **bad), root=tmp0, inputs=[])")
s = s.replace("p, _ = check({k: v for k, v in good.items() if k != \"steps\"}, inputs=[])", "p, _ = check({k: v for k, v in good.items() if k != \"steps\"}, root=tmp0, inputs=[])")
s = s.replace("p, _ = check(good, inputs=[\"no/such/input.npy\"])", "p, _ = check(good, root=tmp0, inputs=[\"no/such/input.npy\"])")
s = s.replace("not check(s, inputs=[])[0]", "not check(s, root=tmp0, inputs=[])[0]")
rep("    import tempfile\n    tmp = tempfile.mkdtemp()\n", "    tmp = tempfile.mkdtemp()\n")
rep('    cfg = json.load(open(a.config, encoding="utf-8"))\n    problems, notes = check(cfg, a.require_clean)', '    cfg = json.load(open(resolve(a.config), encoding="utf-8"))\n    problems, notes = check(cfg, a.require_clean)')
rep("    shutil.rmtree(tmp, ignore_errors=True)\n    print(\"startable selftest passed\")", "    shutil.rmtree(tmp, ignore_errors=True)\n    shutil.rmtree(tmp0, ignore_errors=True)\n    print(\"startable selftest passed\")")
open(p, "w", encoding="utf-8").write(s)
q = "training/train/run_config.py"
t = open(q, encoding="utf-8").read()
a = '    cfg = json.load(open(sys.argv[1], encoding="utf-8"))\n    import startable as ST\n'
assert t.count(a) == 1
t = t.replace(a, '    import startable as ST\n    cfg = json.load(open(ST.resolve(sys.argv[1]), encoding="utf-8"))\n')
open(q, "w", encoding="utf-8").write(t)
print("ok")
