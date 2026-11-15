"""scratch: planted bugs in the critic-round fixes (one-frame 'all', repair hint, BUILT.json marker, startable marker check, camera controls for lane_img_old, interface shape rule); each runs the test that guards it
and must make it fail. Runs on a private copy (cwd = ~/mutk/bev_fusion); files restored in a finally block (backups in temp/t22/mut_backup_k2/)."""
import os
import shutil
import subprocess

FC, LB, ST, IF = "training/feature_cache.py", "training/lane_branch.py", "training/startable.py", "training/interface.py"
BA, BM, BP = "evaluation/bev_adapter.py", "evaluation/bev_modules.py", "evaluation/bev_probe.py"
FILES = (FC, LB, ST, IF, BA, BM, BP)
BK = "temp/t22/mut_backup_k2"
os.makedirs(BK, exist_ok=True)
for f in FILES:
    shutil.copy(f, os.path.join(BK, os.path.basename(f)))
CHAIN = ["tests/test_student_chain.py"]
MUTANTS = [
    ("'all' with a one-frame batch raises again", FC, " and self.cfg.student_frames == \"all\" and Fn > 1 else None", " and self.cfg.student_frames == \"all\" else None", CHAIN),
    ("missing sweep image: the key-frame hint only", FC, "raise FileNotFoundError(f\"{e}; the older frames' images are made by python training/lane_branch.py build-cache-older <config> (into {older_root(self.lane_images)})\") from e", "raise", CHAIN),
    ("lane_images_older defaults to True", FC, "lane_images_older=False, **kw):", "lane_images_older=True, **kw):", CHAIN),
    ("build writes no BUILT.json", LB, "if older and datasets:", "if False:", CHAIN),
    ("BUILT.json key ignores the stride", LB, "return f\"n_cam={n_cam},cam_stride={cam_stride}\"", "return f\"n_cam={n_cam}\"", CHAIN),
    ("second build overwrites the first entry", LB, "built = older_built(root)\n        built[older_key(", "built = {}\n        built[older_key(", CHAIN),
    ("startable does not ask for the marker", ST, "elif older_key(cfg.n_cam, cfg.cam_stride) not in built:", "elif False:", ["training/startable.py"]),
    ("startable asks for the marker of any stride", ST, "elif older_key(cfg.n_cam, cfg.cam_stride) not in built:", "elif not built:", ["training/startable.py"]),
    ("interface: lane_img_old shape rule off", IF, "tuple(batch[\"lane_img_old\"].shape[:2]) != (B, Fn - 1):", "False:", ["training/interface.py"]),
    ("interface: lane_img_old expects F frames", IF, "tuple(batch[\"lane_img_old\"].shape[:2]) != (B, Fn - 1):", "tuple(batch[\"lane_img_old\"].shape[:2]) != (B, Fn):", ["training/interface.py"]),
    ("perturb_camera leaves lane_img_old", BA, "LANE_IMAGE_KEYS = (\"lane_img\", \"lane_img_old\")", "LANE_IMAGE_KEYS = (\"lane_img\",)", CHAIN),
    ("perturb_camera accepts an unknown lane_img* key", BA, "    if unknown:\n        raise ValueError(f\"camera control: item keys", "    if False:\n        raise ValueError(f\"camera control: item keys", CHAIN),
    ("zero_cam leaves lane_img_old", BM, "\"lane_img\", \"lane_img_old\", \"img\") if k in batch", "\"lane_img\", \"img\") if k in batch", CHAIN),
    ("zero_cam selftest does not know lane_img_old", BM, "c[\"lane_img\"].sum() == 0 and c[\"lane_img_old\"].sum() == 0", "c[\"lane_img\"].sum() == 0", ["evaluation/bev_modules.py", "--selftest"]),
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
        failed = [ln.strip() for ln in r.stdout.splitlines() if "FAIL" in ln][:2]
        if r.returncode and not failed and r.stderr.strip():
            failed = [r.stderr.strip().splitlines()[-1]]
        print(("KILLED   " if r.returncode else "SURVIVED ") + name + ("  <- " + " | ".join(failed)[:200] if failed else ""), flush=True)
        if not r.returncode:
            survivors.append(name)
finally:
    for f in FILES:
        shutil.copy(os.path.join(BK, os.path.basename(f)), f)
print("survivors:", survivors or "none")
