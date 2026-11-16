"""CPU smoke of the real jmaps trainer with the E339 teacher (the trainer defaults to cuda: patch the device). usage (project folder): wsl bash -c "nice bash tools/env/_py14.sh temp/_session_scratch_s292/smoke_distill.py"
Runs 20 steps of a d = 32 student against the E307 head of seed 0 (d = 64), then reads the log for the distillation terms."""
import os
import sys

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "training/train"))
import torch  # noqa: E402

torch.set_num_threads(4)
import trainer as T  # noqa: E402

_fit = T.fit
T.fit = lambda *a, **k: _fit(*a, **{**k, "device": "cpu"})
import jmaps_train  # noqa: E402

sys.argv = ["jmaps_train.py", "--name", "smoke_e339_s0", "--seed", "0", "--sigma", "1.0", "--d", "32", "--teacher", os.path.join(ROOT, "runs/e307_n_s0/model.pt"), "--smoke", "--workers", "1", "--bs", "4"]
jmaps_train.main()
print(open(os.path.join(ROOT, "runs/smoke_e339_s0/log.txt")).read()[-1500:])
