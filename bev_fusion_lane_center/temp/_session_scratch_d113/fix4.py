import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)


rep('''    s.add_argument("--prefix", action="store_true", help="the cache is a limited build: compare the first frames of the dump only")''',
    '''    s.add_argument("--prefix", action="store_true", help="the cache is a limited build: compare the first frames of the dump only")
    s.add_argument("--write", action="store_true", help="store the result as fixture.json in the cache directory")
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True, help="the manifests (meta + numbers + fixture) of all finished caches under runs/e365_cache as one JSON file (text record, in git)")''')
rep('''        print(("FIXTURE PASS " if ok else "FIXTURE FAIL ") + json.dumps(rep))
        sys.exit(0 if ok else 1)''', '''        print(("FIXTURE PASS " if ok else "FIXTURE FAIL ") + json.dumps(rep))
        if a.write:
            _atomic_json(os.path.join(_abs(a.cache), "fixture.json"), dict(ok=ok, dump=a.dump, prefix=a.prefix, **rep))
        sys.exit(0 if ok else 1)''')
rep('''    elif a.cmd == "same":''', '''    elif a.cmd == "summary":
        out_ = {}
        for d_ in sorted(glob.glob(os.path.join(CACHE, "*", "*", "manifest.json"))):
            dd = os.path.dirname(d_)
            if "_first" in dd:
                continue
            m_ = json.load(open(d_))
            fx = os.path.join(dd, "fixture.json")
            out_[os.path.relpath(dd, CACHE).replace(os.sep, "/")] = dict(meta={k: v for k, v in m_.items() if k != "numbers"}, numbers=m_["numbers"], fixture=json.load(open(fx)) if os.path.exists(fx) else None)
        _atomic_json(_abs(a.out), out_)
        print(f"{len(out_)} cache directories -> {a.out}")
    elif a.cmd == "same":''')
rep("import argparse\nimport hashlib\n", "import argparse\nimport glob\nimport hashlib\n")
rep("    wsl bash tools/env/_py14.sh lane_tokcache.py same --a", "    wsl bash tools/env/_py14.sh lane_tokcache.py summary --out experiments/E365_e2e-token-champion-s1/cache_manifest.json                          # all manifests + fixtures as one text record\n    wsl bash tools/env/_py14.sh lane_tokcache.py same --a")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
