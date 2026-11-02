"""throw-away: plant bugs in a scratch copy of training/ and check that the CPU tests of the speed work fail for each (BASELINE: unmutated, must pass).
Run from bev_fusion/: ~/miniconda3/envs/gcar/bin/python temp/speed_tools/mutate_host.py"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(".")
WORK = os.path.join(ROOT, "temp", "mut_host", "bev_fusion")
PY = sys.executable
H, S = "test_host_overhead.py", "test_speed_flags.py"

MUTANTS = [
    ("BASELINE host (no change, must PASS)", None, None, None, H),
    ("BASELINE speed (no change, must PASS)", None, None, None, S),
    ("window: last row of the window dropped", "training/object_head.py", "min(X, ix + r + 1)", "min(X, ix + r)", H),
    ("window: first column of the window dropped", "training/object_head.py", "max(0, iy - r), min(Y", "max(1, iy - r), min(Y", H),
    ("window: x and y swapped in the slice", "training/object_head.py", 'w = t["hm"][b, g, x0:x1, y0:y1]', 'w = t["hm"][b, g, y0:y1, x0:x1]', H),
    ("window: peak overwrites instead of max", "training/object_head.py", "w.copy_(torch.maximum(w, peak))", "w.copy_(peak)", H),
    ("one read: back to the read per box", "training/losses.py", "float(w_host[b, ix, iy]) > 0", "float(w_obj[b, 0, ix, iy]) > 0", H),
    ("one read: x and y swapped in the host index", "training/losses.py", "float(w_host[b, ix, iy]) > 0", "float(w_host[b, iy % 120, ix % 104]) > 0", H),
    ("one read: the weight map of sample 0 for all", "training/losses.py", "float(w_host[b, ix, iy]) > 0", "float(w_host[0, ix, iy]) > 0", H),
    ("EMA: the new weights get the decay (multi-tensor path)", "training/train.py", "for _, v in full], alpha=1 - d)", "for _, v in full], alpha=d)", H),
    ("EMA: the new weights get the decay (loop path)", "training/train.py", "self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)", "self.shadow[k].mul_(d).add_(v.detach(), alpha=d)", H),
    ("EMA: frozen keys are averaged too", "training/train.py", "if k not in self.skip]\n        full", "]\n        full", H),
    ("EMA: the shadow is not decayed", "training/train.py", "torch._foreach_mul_(shadow, d)", "torch._foreach_mul_(shadow, 1.0)", H),
    ("EMA: bf16 / fp16 state through the multi-tensor call", "training/train.py", "if v.dtype in (torch.float32, torch.float64)]", "if True]", H),
    ("EMA: no guard for an empty list", "training/train.py", "        if full:  ", "        if True:  ", H),
    ("run_clip: terms narrowed to float32", "training/train.py", "tens.append(v.detach().double().reshape(()))", "tens.append(v.detach().float().reshape(()))", S),
    ("run_clip: the mean over the keyframes forgotten", "training/train.py", "acc[k] = acc.get(k, 0.0) + val / L", "acc[k] = acc.get(k, 0.0) + val", S),
    ("run_clip: the totals not collected", "training/train.py", "                totals.append(val)", "                pass", S),
    ("run_clip: a non-finite keyframe returns nothing", "training/train.py", "            acc, totals = _unpack(kf, L)\n            return acc, False, dg, totals", "            return {}, False, dg, totals", S),
    ("resume: the two switches missing in the check", "training/train.py", 'for k in ("cudnn_benchmark", "tf32") if k not in a and b.get(k)]', 'for k in () if k not in a and b.get(k)]', S),
    ("profiler: no cap on the window", "training/probe.py", "    MAX_STEPS = 8", "    MAX_STEPS = 800", S),
    ("profiler: a failing report raises", "training/probe.py", "        except Exception as e:                                  # noqa: BLE001", "        except KeyError as e:                                  # noqa: BLE001", S),
    ("profiler: a failed report leaves the profiler entered (torch aborts at exit)", "training/probe.py", "            if not self.exited:", "            if False:", S),
    ("profiler: item counted twice again", "training/probe.py", '("aten::item", "cudaStreamSynchronize"', '("aten::item", "aten::_local_scalar_dense", "cudaStreamSynchronize"', S),
]


def fresh():
    if os.path.isdir(os.path.dirname(WORK)):
        shutil.rmtree(os.path.dirname(WORK))
    os.makedirs(os.path.join(WORK, "tests"))
    shutil.copytree("training", os.path.join(WORK, "training"), ignore=shutil.ignore_patterns("__pycache__", "results", "runs"))
    for t in (H, S):
        shutil.copy("tests/" + t, os.path.join(WORK, "tests"))
    os.symlink(os.path.join(ROOT, "..", "autogt"), os.path.join(os.path.dirname(WORK), "autogt"))
    os.symlink(os.path.join(ROOT, "..", "autoresearch"), os.path.join(os.path.dirname(WORK), "autoresearch"))


survived = 0
for name, path, old, new, test in MUTANTS:
    fresh()
    if path:
        p = os.path.join(WORK, path)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
    r = subprocess.run([PY, "tests/" + test], cwd=WORK, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="2"))
    failed = [ln.strip() for ln in r.stdout.splitlines() if "FAIL" in ln and "GATE" not in ln]
    verdict = "PASS" if r.returncode == 0 else "FAIL"
    expect_fail = path is not None
    status = ("killed" if verdict == "FAIL" else "SURVIVED") if expect_fail else ("ok" if verdict == "PASS" else "BASELINE BROKEN")
    if status in ("SURVIVED", "BASELINE BROKEN"):
        survived += 1
    why = failed[0][:90] if failed else ((r.stdout.strip().splitlines() or [""])[-1][:60] + " / " + (r.stderr.strip().splitlines() or [""])[-1][:60] if verdict == "FAIL" else "")
    print(f"{status:16} {name}" + (f"   <- {why}" if why else ""))
print("survivors", survived)
shutil.rmtree(os.path.dirname(WORK))
