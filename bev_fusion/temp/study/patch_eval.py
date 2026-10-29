import re
p = "evaluation/evaluate_run.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''def load_run(run, device):
    with open(os.path.join(run, "config.json")) as f:''', '''def load_run(run, device, ckpt_dir=None):
    with open(os.path.join(run, "config.json")) as f:''')
s = s.replace('''    ckpts = sorted(glob.glob(os.path.join(run, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {run}")
    opt = ''', '''    ckpts = sorted(glob.glob(os.path.join(ckpt_dir or run, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir or run}")
    opt = ''')
s = s.replace('def evaluate_run(run, n=0, prior=0, images=0, timing=False, batch_size=4, workers=4, device="cuda", warm=False, on_train=False):',
 'def evaluate_run(run, n=0, prior=0, images=0, timing=False, batch_size=4, workers=4, device="cuda", warm=False, on_train=False,\n                 control="", ckpt_dir=None):')
s = s.replace('''    whole scene (batch 1; a keyframe whose predecessor is not the previous sample starts with an empty memory); the files
    get the suffix _warm (metrics_warm.json, ...) and the run's cold files stay."""''', '''    whole scene (batch 1; a keyframe whose predecessor is not the previous sample starts with an empty memory); the files
    get the suffix _warm (metrics_warm.json, ...) and the run's cold files stay.
    control (sanity check of what the model really uses, files *_ctl_<name>): shuffle_radar / shuffle_cam = the radar detections /
    camera features of another sample of the batch (cyclic), zero_radar / zero_cam = none. ckpt_dir: read the checkpoint there."""''')
s = s.replace("cfg, model, loss, step, ckpt = load_run(run, device)", "cfg, model, loss, step, ckpt = load_run(run, device, ckpt_dir)")
s = s.replace('''    tag = ("_warm" if warm else "") + ("_train" if on_train else "")           # on_train''', '''    tag = ("_warm" if warm else "") + ("_train" if on_train else "") + (f"_ctl_{control}" if control else "")   # on_train''')
s = s.replace('''        for batch in loader:
            kw = {}
            if warm:                                                  # carry''', '''        for batch in loader:
            kw = {}
            if control:
                perturb(batch, control)
            if warm:                                                  # carry''')
s = s.replace('''# ------------------------------------------------------------------------------------------------ evaluation
def evaluate_run(''', '''# ------------------------------------------------------------------------------------------------ evaluation
def perturb(batch, control):
    """in place: the sanity controls of evaluate_run (the model must lose what the control takes away, else the input is not used)."""
    B = len(batch["tokens"])
    if control == "shuffle_radar":
        batch["radar"]["b"] = (batch["radar"]["b"] + 1) % B
    elif control == "zero_radar":
        batch["radar"] = {k: v[:0] for k, v in batch["radar"].items()}
    elif control == "shuffle_cam":
        batch["prob"], batch["ctx"] = torch.roll(batch["prob"], 1, 0), torch.roll(batch["ctx"], 1, 0)
    elif control == "zero_cam":
        batch["prob"], batch["ctx"] = torch.zeros_like(batch["prob"]), torch.zeros_like(batch["ctx"])
    else:
        raise ValueError(f"unknown control {control}")


def evaluate_run(''', 1)
s = s.replace('''    ap.add_argument("--warm", action="store_true"''', '''    ap.add_argument("--control", default="", choices=["", "shuffle_radar", "zero_radar", "shuffle_cam", "zero_cam"],
                    help="sanity control: perturb one input at evaluation (files *_ctl_<name>)")
    ap.add_argument("--ckpt-dir", default=None, help="read the checkpoint from this folder (e.g. the D: copy of a moved run)")
    ap.add_argument("--warm", action="store_true"''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
