"""scratch: plant bugs in the older-frames change (student_frames 'all') one at a time and run tests/test_student_chain.py; every mutant must make it fail. Files are restored in a finally block (backups in temp/t22/mut_backup_k/)."""
import os
import shutil
import subprocess

FC, LB, TR, MD, ST = "training/feature_cache.py", "training/lane_branch.py", "training/train.py", "training/model.py", "training/startable.py"
FILES = (FC, LB, TR, MD)
BK = "temp/t22/mut_backup_k"
os.makedirs(BK, exist_ok=True)
for f in FILES:
    shutil.copy(f, os.path.join(BK, os.path.basename(f)))

MUTANTS = [
    ("older prob not from the student", FC, "p[:, 1:] if old is None else old[\"prob\" + h].float()", "p[:, 1:]"),
    ("older ctx not from the student", FC, "ctx[:, 1:] if old is None else old[\"ctx\" + h].float()", "ctx[:, 1:]"),
    ("older frames made from the current image", FC, "img = batch[\"lane_img_old\"]\n", "img = batch[\"lane_img\"][:, None].expand(-1, batch[\"lane_img_old\"].shape[1], -1, -1, -1)\n"),
    ("older frames in reversed order", FC, "out = self.student(img.flatten(0, 1))", "out = self.student(img.flip(1).flatten(0, 1))"),
    ("older outputs regrouped k-major", FC, "return {k: v.view(B, K, *v.shape[1:]) for k, v in out.items() if torch.is_tensor(v)}",
     "return {k: v.view(K, B, *v.shape[1:]).transpose(0, 1) for k, v in out.items() if torch.is_tensor(v)}"),
    ("older student pass with a graph", FC, "with torch.no_grad():\n                out = self.student(img.flatten(0, 1))", "with torch.enable_grad():\n                out = self.student(img.flatten(0, 1))"),
    ("count check dropped", FC, "if old is not None and next(iter(old.values())).shape[1] != Fn - 1:", "if False:"),
    ("missing lane_img_old not named", FC, "if \"lane_img_old\" not in batch:\n                raise KeyError(", "if False:\n                raise KeyError("),
    ("dataset: older images in reversed frame order", FC, "for f in item[\"frames\"][1:]])", "for f in item[\"frames\"][1:][::-1]])"),
    ("dataset: older images from the key-frame folder", FC, "load_image(older_root(self.lane_images), f[\"token\"])", "load_image(self.lane_images, f[\"token\"])"),
    ("dataset: older images always loaded", FC, "            if self.lane_images_older:                    # t22", "            if True:                    # t22"),
    ("collate: older images flattened", FC, "torch.stack([it[\"lane_img_old\"] for it in items])", "torch.stack([it[\"lane_img_old\"] for it in items]).flatten(0, 1)"),
    ("dataset: lane_images_older without lane_images accepted", FC, "if lane_images_older and not lane_images:", "if False:"),
    ("make_datasets asks with every student", TR, "bool(cfg.model.get(\"camera_student\")) and cfg.model.get(\"student_frames\", \"current\") == \"all\"", "bool(cfg.model.get(\"camera_student\"))"),
    ("make_datasets never asks", TR, "bool(cfg.model.get(\"camera_student\")) and cfg.model.get(\"student_frames\", \"current\") == \"all\"", "False"),
    ("n_cam rule of the findings dropped", TR, "mcfg.student_frames == \"all\" and cfg.n_cam < 2", "mcfg.student_frames == \"all\" and cfg.n_cam < 1"),
    ("'all' without a student accepted", MD, "if cfg.student_frames == \"all\" and not cfg.camera_student:", "if False:"),
    ("older_records ignores the stride", LB, "track[max(0, i0 - k * ds.cam_stride)] for k in range(1, ds.n_cam)", "track[max(0, i0 - k)] for k in range(1, ds.n_cam)"),
    ("older_records starts at the current frame", LB, "for k in range(1, ds.n_cam)]", "for k in range(0, ds.n_cam)]"),
    ("older cache written into the key-frame folder", LB, "p = image_path(older_root(root), sd[\"token\"])", "p = image_path(root, sd[\"token\"])"),
]
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
py = os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
survivors = []
try:
    for name, path, old, new in MUTANTS:
        s = open(path, encoding="utf-8").read()
        if s.count(old) != 1:
            print(f"SKIP {name}: pattern found {s.count(old)} times", flush=True)
            continue
        open(path, "w", encoding="utf-8").write(s.replace(old, new))
        try:
            r = subprocess.run([py, "tests/test_student_chain.py"], env=env, capture_output=True, text=True, timeout=900)
        finally:
            shutil.copy(os.path.join(BK, os.path.basename(path)), path)
        out = r.stdout.splitlines()
        failed = [ln.strip() for ln in out if "FAIL" in ln][:2]
        if r.returncode and not failed and r.stderr.strip():
            failed = [r.stderr.strip().splitlines()[-1]]
        print(("KILLED   " if r.returncode else "SURVIVED ") + name + ("  <- " + " | ".join(failed)[:240] if failed else ""), flush=True)
        if not r.returncode:
            survivors.append(name)
finally:
    for f in FILES:
        shutil.copy(os.path.join(BK, os.path.basename(f)), f)
print("survivors:", survivors or "none")
