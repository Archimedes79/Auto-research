"""The two training stages of a lane run, from the run's config (the worker composes it): stage 1 trains the token head on the frozen lane
head B' from the token cache (jtoken_train), stage 2 trains the head and B''s node head together (jlive_train). Each stage writes its own
folder below the run (<run>/stage1, <run>/stage2; the trainers keep their own config.json and log there), and stage 2's model.pt is copied,
byte for byte, into the run's folder, where the step `predict` reads it.

    python training/train/lane_train.py stage1|stage2 <the run's config.json>

Config: the blocks `stage1` and `stage2` are the options of the two trainers, key by key (`options`: --<key with dashes> <value>; null = the
trainer's own default; a text may hold {seed}), so an arm of a card sets every one of them through the base config. What the run gives is in
no block: --name (<run>/<stage>), --seed, stage 1's --head-seed (seed + `head_seed_offset`: the champion's 2 / 3 for seeds 0 / 1) and stage
2's --head-run where its block leaves it null (`source`: the stage 1 of `from_run`, else the run's own). `from_run` (a card's `needs`) = a
run below the run root whose finished stage 1 this run starts from instead of training one. A stop request of the worker ends a stage only
through its exit (the trainers watch their own folder); a stage that is complete is not run again. Test: net/tests/test_lane_train.py"""
import json
import os
import shutil
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/train", "../.."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import bflc_paths as BP  # noqa: E402


def options(block, seed):
    """a config block as a trainer's options: every key becomes --<key with dashes> <value>; null is not passed (the trainer's own default);
    a text may hold {seed}"""
    argv = []
    for key, value in block.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise SystemExit(f"config key {key!r}: {value!r} is no value of an option (a number, a text or null)")
        argv += ["--" + key.replace("_", "-"), value.format(seed=seed) if isinstance(value, str) else str(value)]
    return argv


def source(cfg):
    """the run, as a path below the run root, whose stage 1 this run starts from: `from_run`, else the run itself"""
    if not cfg["from_run"]:
        return cfg["name"]
    rel = os.path.relpath(BP.resolve(cfg["from_run"]), BP.RUNS).replace(os.sep, "/")
    if rel == "." or rel.startswith(".."):
        raise SystemExit(f"from_run {cfg['from_run']} is not a run below the run root {BP.RUNS}")
    return rel


def stage_args(cfg, stage):
    """the argv of a stage's trainer: what the run gives (module docstring), then the stage's config block (`options`)"""
    block = dict(cfg[stage])
    own = ["--name", f"{cfg['name']}/{stage}", "--seed", str(cfg["seed"])]
    if stage == "stage1":
        own += ["--head-seed", str(cfg["seed"] + int(block.pop("head_seed_offset")))]
    elif block["head_run"] is None:
        block["head_run"] = f"{source(cfg)}/stage1"
    elif cfg["from_run"]:
        raise SystemExit(f"stage2.head_run {block['head_run']} and from_run {cfg['from_run']} both name the head of stage 1: set one")
    return own + options(block, cfg["seed"])


def main(argv):
    stage, cfg = argv[0], json.load(open(argv[1], encoding="utf-8"))
    if os.path.normpath(cfg["out_root"]) != os.path.normpath(BP.RUNS):
        raise SystemExit(f"the run's out_root {cfg['out_root']} is not the problem's run root {BP.RUNS}")
    run_dir = os.path.join(BP.RUNS, cfg["name"])
    complete = lambda run, s: os.path.isfile(os.path.join(BP.RUNS, run, s, ".done"))    # the trainer's flag: a model.pt alone may be unchecked (a failed export)
    if stage == "stage1":
        src = source(cfg)
        if complete(src, "stage1"):
            print(f"stage 1 of {src} is complete")
            return 0
        if src != cfg["name"]:
            raise SystemExit(f"from_run {src}: its stage 1 is not complete")
        import jtoken_train
        return jtoken_train.main(stage_args(cfg, "stage1")) or 0
    if stage == "stage2":
        if not complete(cfg["name"], "stage2"):
            import jlive_train
            code = jlive_train.main(stage_args(cfg, "stage2")) or 0
            if code:
                return code
        shutil.copyfile(os.path.join(run_dir, "stage2", "model.pt"), os.path.join(run_dir, "model.pt.part"))
        os.replace(os.path.join(run_dir, "model.pt.part"), os.path.join(run_dir, "model.pt"))
        return 0
    raise SystemExit(f"stage {stage!r}: stage1 or stage2")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
