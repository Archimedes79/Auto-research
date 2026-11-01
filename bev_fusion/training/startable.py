"""Is this config startable? Static checks (no GPU, no data read), the smoke config, and the start lines.

    python training/startable.py <config.json>                    check, then print the start lines
    python training/startable.py <config.json> --write-smoke DIR  also write DIR/smoke_<name>.json (20 steps, evaluation and checkpoint
                                                                  every 10, one checkpoint): the end-to-end smoke of a new version
    python training/startable.py                                  self-test

The checks: the file parses (an unknown key is an error), `ModelConfig` / `LossConfig` accept their fields, `model.check_config`
accepts the slot kinds, every path the config names exists (data root, caches, pre-trained weights, `init_from`), the run folder is not
a finished or half-finished run of the same name (it would resume or collide), the disk has room for the checkpoints, and which
uncommitted files of `training/` the launcher's snapshot (a copy of the working tree) will carry. Findings are problems (exit 1)
or notes (exit 0). Run from `bev_fusion/` in the WSL with the gcar environment.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # bev_fusion/
sys.path.insert(0, HERE)

import losses                                                   # noqa: E402
import model as M                                               # noqa: E402
import train as T                                               # noqa: E402

SMOKE = dict(steps=20, warmup=5, eval_every=10, save_every=10, keep=1, log_every=5)
CKPT_GB = 0.45                                                 # one complete checkpoint (state + ema + Adam moments), measured 445.7 MB


def _exists(p, root):
    p = os.path.expanduser(p)
    return os.path.exists(p if os.path.isabs(p) else os.path.join(root, p))


def check(path, root=ROOT):
    """(problems, notes) of the config file `path`; nothing is read from the data and no model is built."""
    problems, notes = [], []
    try:
        cfg = T.read_config(path)
    except Exception as e:                                      # noqa: BLE001 - the message is the finding
        return [f"config does not parse: {e}"], notes
    try:
        mc = M.ModelConfig(**cfg.model)
        losses.LossConfig(**cfg.loss)
        M.check_config(mc)
    except Exception as e:                                      # noqa: BLE001
        problems.append(f"model / loss fields: {e}")
        mc = None
    named = [("data_root", cfg.data_root), ("feature_cache", cfg.feature_cache), ("feature_cache2", cfg.feature_cache2),
             ("gt_cache", cfg.gt_cache), ("radar_cache", cfg.radar_cache), ("lane_images", cfg.lane_images),
             ("init_from", cfg.init_from)]
    if mc is not None:
        named += [("model.trunk_pretrained", mc.trunk_pretrained), ("model.pretrained", mc.pretrained if mc.camera_features == "dualbev" else ""),
                  ("model.lane_pretrained", mc.lane_pretrained if mc.lane_branch == "res4" else "")]
        if mc.camera_features == "dual" and not (cfg.feature_cache and cfg.feature_cache2):
            problems.append("camera_features 'dual' needs feature_cache and feature_cache2")
        if mc.camera_features in ("bevfusion", "dualbev") and cfg.device == "cuda" and not cfg.feature_cache:
            notes.append("no feature_cache: the camera heads run live (slow, much more memory)")
    for key, p in named:
        if p and not _exists(p, root):
            problems.append(f"{key}: {p} does not exist")
    run = os.path.join(root, cfg.out_root, cfg.name)
    if os.path.isdir(run):
        cks = sorted(f for f in os.listdir(run) if f.startswith("ckpt_") and f.endswith(".pt"))
        if os.path.exists(os.path.join(run, ".done")):
            problems.append(f"{cfg.out_root}/{cfg.name} is a finished run (.done): pick another name")
        elif cks:
            notes.append(f"{cfg.out_root}/{cfg.name} has {len(cks)} checkpoint(s), the last {cks[-1]}: starting continues it (a resume, with a new snapshot)")
    free = shutil.disk_usage(root).free / 1e9
    need = CKPT_GB * (cfg.keep + 1)
    if free < need + 5:
        problems.append(f"free disk {free:.1f} GB on the run drive, the checkpoints need about {need:.1f} GB plus margin")
    dirty = _dirty(root)
    if dirty:
        notes.append(f"the snapshot carries {len(dirty)} uncommitted file(s) of training/: {', '.join(dirty[:8])}{' ...' if len(dirty) > 8 else ''}")
    if cfg.steps % cfg.save_every and cfg.steps > cfg.save_every:
        notes.append(f"steps {cfg.steps} is not a multiple of save_every {cfg.save_every}: the last checkpoint is written at the end only")
    return problems, notes


def _dirty(root):
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "training"], cwd=root, capture_output=True, text=True,
                             timeout=30).stdout                   # paths are repo-relative: bev_fusion/training/x.py
    except (OSError, subprocess.SubprocessError):
        return []
    return [line[3:].split("training/", 1)[-1] for line in out.splitlines() if line.strip()]


def smoke_config(path, out_dir):
    """Write the smoke twin of a config: the same model and loss, 20 steps; returns its path."""
    d = json.load(open(path))
    d.update(SMOKE, name="smoke_" + d["name"])
    d["_about"] = f"end-to-end smoke of {d['name'][6:]}: the same model and loss, {SMOKE['steps']} steps, evaluation and a checkpoint every 10"
    d["loss"] = dict(d.get("loss", {}), warmup_hold=0.0, warmup_ramp=0.01)       # the dt != 0 terms and the aux heads run inside 20 steps
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, d["name"] + ".json")
    with open(out, "w") as fh:
        json.dump(d, fh, indent=1)
        fh.write("\n")
    return out


def start_lines(path, name):
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    return (f"AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id {name} --gpu --split train --hyp \"...\" --pred \"...\" "
            f"--rule \"...\" -- bash temp/train/run_ar.sh {rel} > runs/{name}.launcher.log 2>&1 &   (records/README.md: fill hyp / pred / rule from the card)")


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        os.makedirs(os.path.join(tmp, "runs", "done_run"))
        open(os.path.join(tmp, "runs", "done_run", ".done"), "w").close()
        for d in ("data", "cache"):
            os.makedirs(os.path.join(tmp, d))

        def write(**kw):
            c = {"name": "x", "out_root": "runs", "data_root": os.path.join(tmp, "data"), "feature_cache": os.path.join(tmp, "cache"),
                 "gt_cache": "", "model": {"pretrained": ""}, "loss": {}, "device": "cpu"}
            c.update(kw)
            p = os.path.join(tmp, c["name"] + ".json")
            json.dump(c, open(p, "w"))
            return p
        pr, _ = check(write(), tmp)
        assert pr == [], pr
        pr, _ = check(write(bogus_key=1), tmp)
        assert pr and "does not parse" in pr[0], pr
        pr, _ = check(write(model={"no_such_field": 1}), tmp)
        assert any("model / loss fields" in x for x in pr), pr
        pr, _ = check(write(model={"fusion": "no_such_kind"}), tmp)
        assert any("no_such_kind" in x for x in pr), pr
        pr, _ = check(write(loss={"no_such_field": 1}), tmp)
        assert any("model / loss fields" in x for x in pr), pr
        pr, _ = check(write(feature_cache=os.path.join(tmp, "missing")), tmp)
        assert any("feature_cache" in x and "does not exist" in x for x in pr), pr
        pr, _ = check(write(model={"camera_features": "dual", "pretrained": ""}, feature_cache2=""), tmp)
        assert any("needs feature_cache and feature_cache2" in x for x in pr), pr
        pr, _ = check(write(name="done_run"), tmp)
        assert any("finished run" in x for x in pr), pr
        repo = os.path.join(tmp, "repo")                            # uncommitted files of training/ are listed
        os.makedirs(os.path.join(repo, "training"))
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        open(os.path.join(repo, "training", "a.py"), "w").write("x = 1\n")
        for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
            subprocess.run(git + cmd, cwd=repo, check=True, capture_output=True)
        assert _dirty(repo) == [], _dirty(repo)
        open(os.path.join(repo, "training", "a.py"), "w").write("x = 2\n")
        open(os.path.join(repo, "training", "b.py"), "w").write("y = 1\n")
        assert sorted(_dirty(repo)) == ["a.py", "b.py"], _dirty(repo)
        s = smoke_config(write(name="base", steps=6000, loss={"w_obj_vel": 1.0, "fusion_area_only": True}), os.path.join(tmp, "smk"))
        d = json.load(open(s))
        assert d["name"] == "smoke_base" and d["steps"] == 20 and d["loss"]["fusion_area_only"] and d["loss"]["warmup_hold"] == 0.0
        assert T.read_config(s).steps == 20, "the smoke config is itself a valid config"
    print("startable selftest passed: 7 kinds of bad config are found, a good one is clean, uncommitted files are listed, "
          "the smoke twin keeps model and loss")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        _selftest()
        sys.exit(0)
    cfg_path = os.path.abspath(args[0])
    problems, notes = check(cfg_path)
    for n in notes:
        print("note   :", n)
    for p in problems:
        print("PROBLEM:", p)
    name = json.load(open(cfg_path))["name"]
    if "--write-smoke" in sys.argv:
        smk = smoke_config(cfg_path, os.path.abspath(sys.argv[sys.argv.index("--write-smoke") + 1]))
        print("smoke config:", smk)
        print("smoke start :", start_lines(smk, "smoke." + name[:20]))
    print("start       :", start_lines(cfg_path, name.replace("_", ".", 1)))      # run id = config name, first "_" a "." (t10_dual_long_s0 -> t10.dual_long_s0)
    print("startable   :", "NO" if problems else "yes (static checks; the smoke run is the end-to-end proof)")
    sys.exit(1 if problems else 0)
