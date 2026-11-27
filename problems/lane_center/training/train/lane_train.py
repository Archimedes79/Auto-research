"""The two training stages of a lane run, from the run's config (the worker composes it): stage 1 trains the token head on the frozen lane
head B' from the token cache (jtoken_train), stage 2 trains the head and B''s node head together (jlive_train). Each stage writes its own
folder below the run (<run>/stage1, <run>/stage2; the trainers keep their own config.json and log there), and stage 2's model.pt is copied,
byte for byte, into the run's folder, where the step `predict` reads it.

    python training/train/lane_train.py stage1|stage2 <the run's config.json>

Config: stage1 = {ident, head_seed_offset (head seed = seed + offset: the champion's 2 / 3 for seeds 0 / 1), none_weight, steps, bs, lr,
workers}; stage2 = {train_nodes, live_inputs, steps, bs, lr, workers}. A stop request of the worker ends a stage only through its exit
(the trainers watch their own folder); a stage that is complete is not run again."""
import json
import os
import shutil
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/train", "../.."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import bflc_paths as BP  # noqa: E402


def stage1_args(cfg):
    s = cfg["stage1"]
    return ["--name", f"{cfg['name']}/stage1", "--seed", str(cfg["seed"]), "--head-seed", str(cfg["seed"] + int(s["head_seed_offset"])),
            "--ident", s["ident"], "--none-weight", str(s["none_weight"]), "--steps", str(s["steps"]), "--bs", str(s["bs"]), "--lr", str(s["lr"]),
            "--workers", str(s["workers"])]


def stage2_args(cfg):
    s = cfg["stage2"]
    return ["--name", f"{cfg['name']}/stage2", "--seed", str(cfg["seed"]), "--head-run", f"{cfg['name']}/stage1", "--train-nodes", str(s["train_nodes"]),
            "--live-inputs", str(s["live_inputs"]), "--steps", str(s["steps"]), "--bs", str(s["bs"]), "--lr", str(s["lr"]), "--workers", str(s["workers"])]


def main(argv):
    stage, cfg = argv[0], json.load(open(argv[1], encoding="utf-8"))
    if os.path.normpath(cfg["out_root"]) != os.path.normpath(BP.RUNS):
        raise SystemExit(f"the run's out_root {cfg['out_root']} is not the problem's run root {BP.RUNS}")
    run_dir = os.path.join(BP.RUNS, cfg["name"])
    complete = lambda s: os.path.isfile(os.path.join(run_dir, s, ".done"))          # the trainer's flag: a model.pt alone may be unchecked (a failed export)
    if stage == "stage1":
        if complete("stage1"):
            print("stage 1 is complete")
            return 0
        import jtoken_train
        return jtoken_train.main(stage1_args(cfg)) or 0
    if stage == "stage2":
        if not complete("stage2"):
            import jlive_train
            code = jlive_train.main(stage2_args(cfg)) or 0
            if code:
                return code
        shutil.copyfile(os.path.join(run_dir, "stage2", "model.pt"), os.path.join(run_dir, "model.pt.part"))
        os.replace(os.path.join(run_dir, "model.pt.part"), os.path.join(run_dir, "model.pt"))
        return 0
    raise SystemExit(f"stage {stage!r}: stage1 or stage2")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
