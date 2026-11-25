"""Fine-tune of a stored junction system (E324 s) with its maps head physically narrowed (E353, D99 step 1); start it through `autoresearch/tools/run.py`. `jends_train.py` (frozen, E343 digest) adapted: the same e2e stage and loss, the start is the FULL stored state.

    wsl bash tools/env/_py14.sh jnarrow_train.py --name e353_n1_s0 --init runs/e324_s_s0/model.pt --seed 0 --narrow temp/e353/spec_n1_s0.json --steps 1500 --lr 3e-4 --e2e --scales --maps-w 1.0 --geom-w 0.5 --attach --workers 4 [--smoke]
    ... the same without --narrow (or with a spec without edits)       # the equal-budget control cp: no surgery
    ... --steps 0                                                      # the narrowed model before any training (n1z): runs/<name>/model.pt, no step
    wsl bash tools/env/_py14.sh jnarrow_train.py dump | score | time [paired] ...   # the evaluation path of the result (`jnarrow_eval.py`; the exact calls are in its docstring)

--narrow is the JSON spec of `jnarrow.py` ({"calib": path, "edits": [{"unit" | "tap": name, "frac": removed share}]}). The run folder keeps the resolved spec (`narrow.json`) and a copy of the calibration file (`calib.pt`, its sha256 in the spec); a folder that holds another spec refuses to resume.
model.pt = the narrowed state + the args of the init run with `narrow` (the resolved spec), `init`, `init_run`, `seed` ...: `jends_eval.py score` reads it as it is, `dump` / `jtiming.py` through `jnarrow_eval.py`. --e2e, --scales, --attach must match the init run (they are its architecture and data flags; --attach needs --geom-w > 0);
--maps-w, --geom-w, --cross-w, --driven-w, --raster default to the init run's values (the E324 s loss). Trainable: the pair head and the maps head (the new bias included), the pyramid and B's node head stay frozen. The seed (default: the init run's) is the one `jends_eval.py score` asserts; the data order is the same for
every arm of a seed (`trainer.fit` re-seeds after the model is built). Checks at the start: the 57 tensors of the pyramid, the node head, `BFLC_SIG` against the sigma of the maps run (`jends_train.py` leaves that to the launcher), the seed against the maps path, the run folder (`check_name`: never a stored run's folder; a folder with a
model.pt or ckpt.pt is only resumed when its args show the same init run and seed). The log carries the MEASURED GMACs of the init and of the run's model. The maps head of a narrowed run is not written as a plain maps run (`<name>_maps`): its shapes are not those of `JModel`.
"""
import argparse
import json
import os
import re
import shutil
import sys

import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jends as JE  # noqa: E402
import loss_jends as LJE  # noqa: E402
import jgt as JG  # noqa: E402
import jmaps as JM  # noqa: E402
import loss_jmaps as LJM  # noqa: E402
import jnarrow as JN  # noqa: E402
import jprune_probe as PP  # noqa: E402
import lane_gt_data as LD  # noqa: E402
import trainer as T  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", required=True, help="runs/<run>/model.pt: the FULL stored state of a pair run (the maps head and the pyramid included)")
    ap.add_argument("--narrow", default=None, help="JSON spec of jnarrow.py; none or without edits = no surgery (the control)")
    ap.add_argument("--seed", type=int, default=None, help="default: the init run's seed")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--e2e", action="store_true", help="required: the e2e stage of E324 s (the pair loss reaches the maps head)")
    ap.add_argument("--scales", action="store_true", help="must match the init run")
    ap.add_argument("--attach", action="store_true", help="must match the init run")
    ap.add_argument("--maps-w", type=float, default=None)
    ap.add_argument("--geom-w", type=float, default=None)
    ap.add_argument("--cross-w", type=float, default=None)
    ap.add_argument("--driven-w", type=float, default=None)
    ap.add_argument("--raster", default=None, choices=("legacy", "exact"))
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="20 steps")
    return ap.parse_args(argv)


def run_of(path):
    """the run name of `runs/<run>/model.pt` (the loaders read a run by its name)"""
    p = os.path.normpath(PP.rootpath(path))
    run = os.path.basename(os.path.dirname(p))
    assert p == os.path.normpath(os.path.join(JN.JEV.ROOT, "runs", run, "model.pt")), f"--init must be runs/<run>/model.pt, not {path}"
    return run


def load_spec(path):
    spec = json.load(open(PP.rootpath(path))) if path else {}
    return spec if isinstance(spec, dict) and spec else {}


def check_init(model, args0, a):
    """what the start must be: a pair run with the pyramid (57 tensors) and, for the attachment, B's node head; the flags of the architecture and the data flow; the seed of its maps run; the width of the end Gaussians"""
    assert a.e2e, "--e2e is required: this trainer is the e2e stage of E324 s"
    assert not args0.get("narrow", {}).get("taps"), "--init is a narrowed run: narrow the stored run instead"
    assert a.scales == bool(args0.get("scales")) and a.attach == bool(args0.get("attach")), f"--scales / --attach must match the init run (scales {args0.get('scales')}, attach {args0.get('attach')})"
    assert not a.attach or a.geom_w > 0, "--attach needs --geom-w > 0 (the attachment is trained by the geometry loss on the final curve and its cross entropy)"
    keys = model.state_dict()
    assert sum(k.startswith("jmodel.pyr.") for k in keys) == 57, "the pyramid of the init run must have its 57 tensors"
    assert (model.nodes is not None) == bool(args0.get("attach") or args0.get("ntok")), "B's node head must be in the model exactly when the run attaches / reads node tokens"
    m = re.search(r"_s(\d+)[/\\]", str(args0.get("maps", "")).replace("\\", "/") + "/")
    assert m is None or int(m.group(1)) == a.seed, f"--seed {a.seed} does not match the maps run of the init ({args0.get('maps')}): `jends_eval.py score` would refuse the result"
    want, sig = args0["maps_args"].get("sigma"), float(os.environ.get("BFLC_SIG", JG.SIG))
    assert want is None or abs(sig - float(want)) < 1e-9, f"BFLC_SIG={sig} but the maps run was trained with sigma {want}: export BFLC_SIG={want} (the heat targets would not match the maps)"


def set_trainable(model):
    """the e2e stage of the original: the pair head and the maps head train, the pyramid and B's node head are frozen -> the parameters that train"""
    ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    return ps


def record_narrowing(run, spec):
    """the resolved spec and a copy of its calibration file in the run folder; a folder that holds another spec is not resumed"""
    want, p = json.loads(json.dumps(spec)), run.path("narrow.json")
    if os.path.exists(p):
        assert json.load(open(p)) == want, f"{p} holds another spec: a resumed run keeps its spec (use another --name)"
        return
    if spec["taps"]:
        shutil.copyfile(PP.rootpath(spec["calib"]), run.path("calib.pt"))
        assert JN.sha256_file(run.path("calib.pt")) == spec["calib_sha256"]
    json.dump(want, open(p, "w"), indent=1)


def stored_args(folder):
    """the args a run folder was written with: its model.pt (a finished run) or its config.json (a running or stopped one); None when it holds neither"""
    p = os.path.join(folder, "model.pt")
    if os.path.exists(p):
        return torch.load(p, map_location="cpu")["args"]
    p = os.path.join(folder, "config.json")
    return json.load(open(p)).get("args") if os.path.exists(p) else None


def check_name(name, folder, run_init, seed):
    """the folder of a new arm is never a stored run's: `--name` is not the init run, and a folder that holds a model.pt or a ckpt.pt is only RESUMED, i.e. its args show the same init run and seed (a typo `--name e324_s_s0` would resume the stored step-6000
    checkpoint, rewrite its model.pt and write the spec into it)"""
    assert name != run_init, f"--name {name} is the init run: its folder is a stored run"
    held = [f for f in ("model.pt", "ckpt.pt") if os.path.exists(os.path.join(folder, f))]
    if held:
        a0 = stored_args(folder) or {}
        assert a0.get("init_run") == run_init and a0.get("seed") == seed, f"{folder} holds {held} of another run (init {a0.get('init_run')}, seed {a0.get('seed')}), not of init {run_init} seed {seed}: use another --name"


def macs_line(m0, m1):
    return " ".join(f"{k} {m0[k]:.4f} -> {m1[k]:.4f}" for k in ("pyr", "maps", "pair", "total")) + f", saved {m0['total'] - m1['total']:.4f}"


def make_loss(a, run):
    """the loss of `jends_train.main` (pair loss + the maps loss of --e2e), a copy of its closure (the test compares the two)"""
    calls = [0]

    def loss_fn(m, b):
        dev = next(m.parameters()).device
        x = b["x"].to(dev, non_blocking=True).float()
        gt = {k: b[k].to(dev) for k in ("ent", "ext", "ent_m", "ext_m")}
        _, jout, out = m(x, gt=gt)
        loss, info = LJE.jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w, geom_w=a.geom_w)
        calls[0] += 1
        if a.geom_w > 0 and calls[0] == 100:                                                              # critic 2026-11-12: a wrong pair alignment of the GT polylines shows as geom_n far below pos_pairs
            run.log(f"step 100 check of the geometry loss: geom_n {info.get('geom_n', 0.0):.0f} pairs with a GT polyline of pos_pairs {info.get('pos_pairs', 0.0):.0f} positives, geom_end_d {info.get('geom_end_d', float('nan')):.2f} m" + (f", attachment over {info['att_n']:.0f} keypoints with a candidate: att_acc {info['att_acc']:.2f} (nearest-candidate baseline {info['att_acc_nearest']:.2f}), att_ce {info['att_ce']:.2f}, attached end error {info['att_d']:.2f} m against the keypoint's {info['att_kp_d']:.2f} m" if "att_ce" in info else ""))
        lm, im = LJM.jmaps_loss(jout, b)
        loss = loss + a.maps_w * lm
        info.update({"j_" + k: v for k, v in im.items()})
        return loss, info
    return loss_fn


def main(argv=None):
    a = parse_args(argv)
    init = PP.rootpath(a.init)
    run_init = run_of(init)
    model, args0 = JN.load_model(run_init, "cpu")
    a.seed = args0["seed"] if a.seed is None else a.seed
    for k in ("maps_w", "geom_w", "cross_w", "driven_w", "raster"):                                       # the loss of the init run unless the command line says otherwise
        if getattr(a, k) is None:
            assert k in args0, f"the init run does not store {k}: pass --{k.replace('_', '-')}"
            setattr(a, k, args0[k])
    check_init(model, args0, a)
    run = T.Run(a.name)
    check_name(a.name, run.dir, run_init, a.seed)                                                         # before anything is written into the folder
    if a.raster == "exact":
        os.environ["BFLC_RASTER"] = "exact"
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PSCALE = a.scales                                                                   # the fitted handle scales cost 0.15 s per frame: only for the scale head
    LD.QData.DEFAULT_PPTS = a.geom_w > 0                                                                 # the GT connector polylines of the true pairs, only for the geometry loss
    macs0 = JN.measured_macs(model)
    spec = JN.resolve(model, load_spec(a.narrow))
    JN.narrow(model, spec)
    macs1 = JN.measured_macs(model) if model.narrow_spec["taps"] else macs0                              # no surgery: the same model
    model.e2e = True                                                                                     # the maps head gets the gradient of the pair loss (the pyramid stays under no_grad in `JModel.forward`)
    record_narrowing(run, model.narrow_spec)
    steps = 20 if a.smoke else a.steps
    ps = set_trainable(model)
    args = dict(args0, name=a.name, seed=a.seed, init=a.init, init_run=run_init, narrow=model.narrow_spec, steps=steps, lr=a.lr, bs=a.bs, workers=a.workers, smoke=a.smoke, e2e=True, maps_w=a.maps_w, geom_w=a.geom_w, cross_w=a.cross_w, driven_w=a.driven_w, raster=a.raster)
    run.log(f"jnarrow init {run_init} narrow {[(t, len(v['removed'])) for t, v in model.narrow_spec['taps'].items()] or 'none'}: {steps} steps, lr {a.lr}, trainable {sum(p.numel() for p in ps) / 1e6:.3f} M of {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M, e2e, geom_w {a.geom_w}, attach {a.attach}")
    run.log(f"measured GMACs per frame (FlopCounterMode, batch 1; init -> this run): {macs_line(macs0, macs1)}")
    if steps:
        ds, dl = LD.loader("train", bs=a.bs, workers=a.workers, shuffle=True, map_mode="train", pixels=False)
        run.log(f"train frames {len(ds)} (missing lane GT: {ds.missing})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T.fit(model, make_loss(a, run), run, steps=steps, loader=dl if steps else None, batch_fn=None if steps else (lambda step: None), lr=a.lr, ckpt_every=500, seed=a.seed, args=args, params=ps, device=dev,
          config=dict(design="jends", init=a.init, narrow=model.narrow_spec, e2e=True, scales=a.scales, attach=a.attach, geom_w=a.geom_w, maps=args0.get("maps")))
    run.log("done", a.name)


if __name__ == "__main__":
    if sys.argv[1:2] and sys.argv[1] in ("dump", "score", "time"):
        import jnarrow_eval as JNE
        JNE.main(sys.argv[1:])
    else:
        main()
