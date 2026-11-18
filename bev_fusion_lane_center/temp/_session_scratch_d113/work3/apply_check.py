import importlib.util, os, sys
R = os.environ["BFLC_ROOT"]
def chk(testfile, srcmap):
    spec = importlib.util.spec_from_file_location("t", testfile)
    t = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(t)
    bad = 0
    for m in t.MUTANTS:
        if len(m) >= 4 and testfile.endswith("test_jmap_train.py") and m[0] in srcmap:
            src = open(srcmap[m[0]], encoding="utf-8").read(); a = m[2]
        elif testfile.endswith("test_jmap_train.py"):
            continue
        else:
            src = open(srcmap["*"], encoding="utf-8").read(); a = m[1]
        if testfile.endswith("test_jmap_train.py"):
            a = m[2]
        c = src.count(a)
        if c != 1:
            bad += 1
            print("NOT APPLIED", c, m[0] if testfile.endswith("train.py") else m[0], "|", (m[1] if not testfile.endswith("train.py") else m[1])[:60])
    print(testfile.split("/")[-1], len(t.MUTANTS), "mutants,", bad, "not applied")
chk(os.path.join(R, "training/grid/test_lane_map.py"), {"*": os.path.join(R, "training/data/lane_map_data.py")})
chk(os.path.join(R, "training/grid/test_jmap.py"), {"*": os.path.join(R, "training/grid/jmap.py")})
chk(os.path.join(R, "training/grid/test_jmap_train.py"), {"jmap_train.py": os.path.join(R, "training/train/jmap_train.py"), "jmap_eval.py": os.path.join(R, "evaluation/jmap_eval.py")})
