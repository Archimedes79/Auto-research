"""scratch: the stage-A mutants whose code the stage-B step MOVED (the loss to student_encoder.py, the eval-only loader to student_encoder.load_trained / newest_checkpoint): plant them where the code lives now and run tests/test_student_encoder.py (WSL, from bev_fusion/)"""
import os, shutil, subprocess, sys
SRC, MUT = os.path.abspath("."), os.path.abspath("temp/t22/mut_moved")
SE, DS = "student_encoder.py", "distill_student.py"
M = [
 (SE, 'm["kl" + k] = (t * (t.clamp_min(1e-8).log() - out["logit" + k].float().log_softmax(1))).sum(1)', 'lp = out["logit" + k].float().log_softmax(1)\n        m["kl" + k] = (lp.exp() * (lp - t.clamp_min(1e-8).log())).sum(1)', "KL reversed"),
 (SE, "        t = t / t.sum(1, keepdim=True)\n", "", "teacher not renormalised"),
 (SE, 'mse" + k] = (c - tc).pow(2).mean(1)', 'mse" + k] = (c - tc).pow(2).sum(1)', "mse summed"),
 (SE, "names[-1]) if names else None", "names[0]) if names else None", "eval-only takes the oldest checkpoint"),
 (SE, "    fresh = warm_start(file, model, nn.Module(), ck=ck)", '    model.load_state_dict({k[6:]: v for k, v in ck["state"].items() if k.startswith("model.")}, strict=False)\n    fresh = []', "eval-only loads live weights"),
 (DS, "    path = newest_checkpoint(out_dir)\n    if path is None:", "    path = None\n    if path is None:", "eval-only never finds a checkpoint"),
 (DS, 'os.path.join(out_dir, "eval_metrics.json")', 'os.path.join(out_dir, "metrics.json")', "eval-only overwrites metrics.json"),
]
def fresh():
    if os.path.exists(MUT): shutil.rmtree(MUT)
    os.makedirs(MUT)
    shutil.copytree(os.path.join(SRC, "training"), os.path.join(MUT, "training"), ignore=shutil.ignore_patterns("__pycache__", "configs", "legacy"))
    os.makedirs(os.path.join(MUT, "tests")); shutil.copy(os.path.join(SRC, "tests", "test_student_encoder.py"), os.path.join(MUT, "tests"))
    os.symlink(os.path.join(SRC, "models"), os.path.join(MUT, "models"))
env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
def run():
    r = subprocess.run([sys.executable, "tests/test_student_encoder.py"], cwd=MUT, capture_output=True, text=True, env=env)
    return r.returncode, [l.strip() for l in r.stdout.splitlines() if "FAIL" in l][:2] + ([r.stderr.strip().splitlines()[-1]] if r.returncode and not any("FAIL" in l for l in r.stdout.splitlines()) else [])
fresh(); rc, _ = run(); print("baseline exit", rc, flush=True)
alive = []
for f, old, new, name in M:
    fresh()
    p = os.path.join(MUT, "training", f); s = open(p, encoding="utf-8").read()
    if old not in s: print("PATTERN MISSING:", name, flush=True); alive.append(name); continue
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    rc, why = run()
    print(("killed " if rc else "ALIVE  ") + name, "|", (why[0] if why else "")[:150], flush=True)
    if not rc: alive.append(name)
print("alive:", alive)
