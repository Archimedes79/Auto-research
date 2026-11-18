"""scratch launcher: numpy / torch are imported with the 319 project dirs of PYTHONPATH moved behind site-packages (each import otherwise walks them on /mnt/c: 50 s), then the order is restored and the script runs"""
import os, sys, runpy
orig = list(sys.path)
r = os.environ["BFLC_ROOT"]
sys.path[:] = [x for x in orig if not x.startswith(r)] + [x for x in orig if x.startswith(r)]
import numpy, torch  # noqa
sys.path[:] = orig
script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))
runpy.run_path(script, run_name="__main__")
