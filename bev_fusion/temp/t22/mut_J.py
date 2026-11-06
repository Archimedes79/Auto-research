"""scratch: plant bugs in the memory fix one at a time and run tests/test_student_chain.py; every mutant must make it fail. Files are restored in a finally block (backups in temp/t22/mut_backup/)."""
import os
import shutil
import subprocess
import sys

FC, SE = "training/feature_cache.py", "training/student_encoder.py"
os.makedirs("temp/t22/mut_backup", exist_ok=True)
for f in (FC, SE):
    shutil.copy(f, os.path.join("temp/t22/mut_backup", os.path.basename(f)))

MUTANTS = [
    ("wrong p in the recomputed lift", FC,
     "args = (ctx[b], p[b], batch[\"K\"][b, 0]", "args = (ctx[b], p[(b + 1) % B], batch[\"K\"][b, 0]"),
    ("K of another sample in every lift call", FC,
     "batch[\"K\"][b, 0], batch[\"T_cam_car\"][b, 0], batch[\"hw\"]", "batch[\"K\"][(b + 1) % B, 0], batch[\"T_cam_car\"][b, 0], batch[\"hw\"]"),
    ("collapse inside the checkpoint", FC,
     "vols.append(collapse(torch.utils.checkpoint.checkpoint(lift, *args, key=key, use_reentrant=False)))",
     "vols.append(torch.utils.checkpoint.checkpoint(lambda *a, **k: collapse(lift(*a, **k)), *args, key=key, use_reentrant=False))"),
    ("lift recompute also in evaluation mode", FC,
     "recompute_lift = stud is not None and self.cfg.student_trainable and self.training and torch.is_grad_enabled()",
     "recompute_lift = stud is not None and self.cfg.student_trainable and torch.is_grad_enabled()"),
    ("encoder recompute without the BatchNorm guard", SE,
     "if self.checkpoint_encode and self._bn_eval and self.training", "if self.checkpoint_encode and self.training"),
    ("encoder recompute also in evaluation mode", SE,
     "if self.checkpoint_encode and self._bn_eval and self.training and torch.is_grad_enabled()", "if self.checkpoint_encode and self._bn_eval and torch.is_grad_enabled()"),
    ("lift never recomputed", FC,
     "recompute_lift = stud is not None and self.cfg.student_trainable and self.training and torch.is_grad_enabled()", "recompute_lift = False"),
    ("the key= dropped from the checkpointed lift", FC,
     "checkpoint(lift, *args, key=key, use_reentrant=False)", "checkpoint(lift, *args, use_reentrant=False)"),
]
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
py = os.path.expanduser("~/miniconda3/envs/gcar/bin/python")
survivors = []
try:
    for name, path, old, new in MUTANTS:
        s = open(path, encoding="utf-8").read()
        if s.count(old) != 1:
            print(f"SKIP {name}: pattern found {s.count(old)} times")
            continue
        open(path, "w", encoding="utf-8").write(s.replace(old, new))
        try:
            r = subprocess.run([py, "tests/test_student_chain.py"], env=env, capture_output=True, text=True, timeout=600)
        finally:
            shutil.copy(os.path.join("temp/t22/mut_backup", os.path.basename(path)), path)
        failed = [ln.strip() for ln in r.stdout.splitlines() if "FAIL" in ln][:2] + ([r.stderr.strip().splitlines()[-1]] if r.returncode and not any("FAIL" in ln for ln in r.stdout.splitlines()) and r.stderr.strip() else [])
        print(("KILLED   " if r.returncode else "SURVIVED ") + name + ("  <- " + " | ".join(failed)[:260] if failed else ""), flush=True)
        if not r.returncode:
            survivors.append(name)
finally:
    for f in (FC, SE):
        shutil.copy(os.path.join("temp/t22/mut_backup", os.path.basename(f)), f)
print("survivors:", survivors or "none")
