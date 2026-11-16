"""scratch: mutation of jdiag.pooled_boot: one scene draw per SEED instead of one per resample (the test must fail)"""
import os, shutil, subprocess
R = "/mnt/c/Auto-research/bev_fusion_lane_center"
d = f"{R}/temp/_session_scratch_e291/mut/jd1"
shutil.rmtree(d, ignore_errors=True)
shutil.copytree(f"{R}/temp/_session_scratch_e291/stage", d)
s = open(f"{R}/evaluation/jdiag.py", encoding="utf-8").read()
old = "        pick = np.concatenate([where[uniq[i]] for i in rng.randint(0, len(uniq), len(uniq))])\n        seed_d[r] = [diff(A, B, pick) for A, B in data]"
new = "        seed_d[r] = [diff(A, B, np.concatenate([where[uniq[i]] for i in rng.randint(0, len(uniq), len(uniq))])) for A, B in data]"
assert s.count(old) == 1
open(f"{d}/evaluation/jdiag.py", "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
env = dict(os.environ, BFLC_STAGE=d, E291_FAST="1")
p = subprocess.run(["bash", f"{R}/temp/e291/run_staged.sh", "jdiag.py", "selftest"], env=env, capture_output=True, text=True, cwd=R)
errs = [l for l in p.stderr.splitlines() if "Error" in l]
print("KILLED" if p.returncode != 0 else "SURVIVED", "j01_two_draws:", p.returncode, [l for l in p.stderr.splitlines() if l.strip()][-4:])
shutil.rmtree(d)
