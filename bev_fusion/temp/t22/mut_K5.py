"""scratch: the mutants of mut_K2 whose first run was invalid (the private copy lacked autoresearch/evalsuite, so the test crashed at the camera-controls block and the mutants "died" of that crash)
plus the key-function mutant whose check compared the key with the key function itself. Runs on the private copy ~/mutk2 (cwd = ~/mutk2/bev_fusion) after the baseline passed there."""
import os
import shutil
import subprocess

LB, BA, BM, BP = "training/lane_branch.py", "evaluation/bev_adapter.py", "evaluation/bev_modules.py", "evaluation/bev_probe.py"
FILES = (LB, BA, BM, BP)
BK = "temp/t22/mut_backup_k5"
os.makedirs(BK, exist_ok=True)
for f in FILES:
    shutil.copy(f, os.path.join(BK, os.path.basename(f)))
CHAIN = ["tests/test_student_chain.py"]
MUTANTS = [
    ("BUILT.json key ignores the stride", LB, "return f\"n_cam={n_cam},cam_stride={cam_stride}\"", "return f\"n_cam={n_cam}\"", CHAIN),
    ("perturb_camera leaves lane_img_old", BA, "LANE_IMAGE_KEYS = (\"lane_img\", \"lane_img_old\")", "LANE_IMAGE_KEYS = (\"lane_img\",)", CHAIN),
    ("perturb_camera accepts an unknown lane_img* key", BA, "    if unknown:\n        raise ValueError(f\"camera control: item keys", "    if False:\n        raise ValueError(f\"camera control: item keys", CHAIN),
    ("zero_cam leaves lane_img_old", BM, "\"lane_img\", \"lane_img_old\", \"img\") if k in batch", "\"lane_img\", \"img\") if k in batch", CHAIN),
    ("probe perturb leaves lane_img_old", BP, "\"img\", \"lane_img\", \"lane_img_old\"):", "\"img\", \"lane_img\"):", CHAIN),
]
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
py = os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
survivors = []
try:
    for name, path, old, new, cmd in MUTANTS:
        s = open(path, encoding="utf-8").read()
        if s.count(old) != 1:
            print(f"SKIP {name}: pattern found {s.count(old)} times", flush=True)
            continue
        open(path, "w", encoding="utf-8").write(s.replace(old, new))
        try:
            r = subprocess.run([py] + cmd, env=env, capture_output=True, text=True, timeout=900)
        finally:
            shutil.copy(os.path.join(BK, os.path.basename(path)), path)
        fails = [ln.strip() for ln in r.stdout.splitlines() if "FAIL" in ln]
        crash = r.stderr.strip().splitlines()[-1] if r.returncode and r.stderr.strip() else ""
        # valid only when a named CHECK failed (a FAIL line), not when the test crashed
        verdict = "KILLED   " if r.returncode and fails else ("CRASHED  (not a valid kill) " if r.returncode else "SURVIVED ")
        print(verdict + name + "  <- " + (" | ".join(fails[:2])[:200] if fails else crash[:160]), flush=True)
        if not (r.returncode and fails):
            survivors.append(name)
finally:
    for f in FILES:
        shutil.copy(os.path.join(BK, os.path.basename(f)), f)
print("not validly killed:", survivors or "none")
