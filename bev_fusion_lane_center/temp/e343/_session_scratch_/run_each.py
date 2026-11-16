import os, sys, traceback
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests"))
import test_jtest_eval as T
names = sys.argv[1:] or sorted(k for k in vars(T) if k.startswith("test_"))
for k in names:
    try:
        getattr(T, k)()
        print("PASS", k, flush=True)
    except BaseException as e:
        print("FAIL", k, type(e).__name__, str(e)[:300].replace("\n", " "), flush=True)
        traceback.print_exc(limit=-4)
