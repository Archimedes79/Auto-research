"""shim (2026-11-12): the scratch maneuver measure moved to evaluation/jmaneuver.py after the critic pass (the old script limited the direction test to the 6 nearest delivered points: kept as maneuver_old_k6.py). usage as before: maneuver.py <run> <seed> [variants]"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
run, seed = sys.argv[1], sys.argv[2]
variants = [("p" + v[3:]) if v.startswith("p>=") else v for v in sys.argv[3:]] or ["plain", "p0.5", "p0.1", "oracle", "oracle+fit"]
sys.exit(subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "jmaneuver.py"), "score", "--run", run, "--seed", seed, "--variants"] + variants + ["--rows"]))
