import os, sys, time
sys.path.insert(0, os.path.join(os.environ['BFLC_ROOT'], 'evaluation'))
import jtest_eval as J
for i in range(3):
    t = time.time(); f = J.code_sha256(); print(len(f), round(time.time() - t, 2), flush=True)
