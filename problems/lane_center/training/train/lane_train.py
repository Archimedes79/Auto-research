"""The training stages of a lane run, from the run's config (the worker composes it): stage 0 trains the lane head B' (qgraph_train: the parts
STAGE0, the blocks of `stage0`; only with stage0.enabled), the step `tokens` builds the token cache of the run's B' into <run>/tokens (TRAIN and VAL,
`lane_tokcache.build`; only with tokens.enabled; B' = the run's stage 0, else the stored runs/<stage1.ident>_s<seed>), stage 1 trains the token head
on the frozen B' from the token cache (jtoken_train; with tokens.enabled its ident is the run's own cache, runs/<run>/tokens, which stage 2, the
predict step and the evaluation then read with the B' it was made from), stage 2 trains the head and B''s node head together (jlive_train). Each
stage writes its own folder below the run (<run>/stage0 with a folder per part, <run>/tokens, <run>/stage1, <run>/stage2), stage 0's last part's
model.pt is copied to <run>/stage0/model.pt, and stage 2's model.pt, byte for byte, into the run's folder, where the step `predict` reads it.
With both blocks off (the base config) the run uses the stored B' and its stored cache.

    python -P -m lane_center.training.train.lane_train stage0|tokens|stage1|stage2 <the run's config.json>

Config: the blocks `stage1` and `stage2` are the options of the two trainers, key by key (`options`: --<key with dashes> <value>; null = the
trainer's own default; a text may hold {seed}), so an arm of a card sets every one of them through the base config. What the run gives is in
no block: --name (<run>/<stage>), --seed, stage 1's --head-seed (seed + `head_seed_offset`: the champion's 2 / 3 for seeds 0 / 1) and stage
2's --head-run where its block leaves it null (`source`: the stage 1 of `from_run`, else the run's own). `from_run` (a card's `needs`) = a
run below the run root whose finished stage 1 this run starts from instead of training one. A stop request of the worker ends a stage only
through its exit (the trainers watch their own folder); a stage that is complete is not run again. Test: net/tests/test_lane_train.py"""
import argparse
import json
import os
import shutil
import sys

from lane_center.training import bflc_paths as BP

STAGE0 = ("e271", "e273", "e292")             # the parts of stage 0 in order (blocks of stage0): the E271 grid head, B (E273 p: phases 2 - 7), B' (E292 a)


def options(block, seed):
    """a config block as a trainer's options: every key becomes --<key with dashes> <value>; a list --<key> <its values> (an empty one: the option alone);
    null is not passed (the trainer's own default); a text may hold {seed}"""
    argv = []
    for key, value in block.items():
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, bool) or not isinstance(v, (int, float, str)) for v in values):
            raise SystemExit(f"config key {key!r}: {value!r} is no value of an option (a number, a text, a list of them or null)")
        argv += ["--" + key.replace("_", "-")] + [v.format(seed=seed) if isinstance(v, str) else str(v) for v in values]
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
        if enabled(cfg, "tokens"):                                                   # the run's own token cache (the step `tokens`) in place of the stored one of `ident`
            block["ident"] = own_tokens(cfg)
    elif block["head_run"] is None:
        block["head_run"] = f"{source(cfg)}/stage1"
    elif cfg["from_run"]:
        raise SystemExit(f"stage2.head_run {block['head_run']} and from_run {cfg['from_run']} both name the head of stage 1: set one")
    return own + options(block, cfg["seed"])


def enabled(cfg, block):
    """is the block `stage0` / `tokens` switched on (a config without the block: off)"""
    return bool((cfg.get(block) or {}).get("enabled", 0))


def own_tokens(cfg):
    """the folder of the run's own token cache (the step `tokens`), in the form stage 1 takes as its `ident`"""
    return f"runs/{cfg['name']}/tokens"


def tokens_b(cfg):
    """the B' of the step `tokens`: the run's own stage 0 when it trains one, else the B' of stage 1's ident (`lane_ptok_data.b_run`: runs/<ident>_s<seed>, or
    the B' of a cache folder)"""
    from lane_center.net.prep import lane_ptok_data as LP
    return f"runs/{cfg['name']}/stage0" if enabled(cfg, "stage0") else LP.b_run(cfg["stage1"]["ident"], cfg["seed"])


def stage0_args(cfg, part):
    """the argv of a part of stage 0 (STAGE0): the part's block (`options`) and what the run gives: --name <run>/stage0/<part>, --seed, the E271 head of the run
    (--head, every part after e271) and the model of the part before (--resume-from, every part after the first B' part)"""
    own = ["--name", f"{cfg['name']}/stage0/{part}", "--seed", str(cfg["seed"])]
    i = STAGE0.index(part)
    if i > 0:
        own += ["--head", f"runs/{cfg['name']}/stage0/{STAGE0[0]}/model.pt"]
    if i > 1:
        own += ["--resume-from", f"runs/{cfg['name']}/stage0/{STAGE0[i - 1]}/model.pt"]
    return own + options(cfg["stage0"][part], cfg["seed"])


def main(argv):
    stage, cfg = argv[0], json.load(open(argv[1], encoding="utf-8"))
    if os.path.normpath(cfg["out_root"]) != os.path.normpath(BP.RUNS):
        raise SystemExit(f"the run's out_root {cfg['out_root']} is not the problem's run root {BP.RUNS}")
    run_dir = os.path.join(BP.RUNS, cfg["name"])
    complete = lambda run, s: os.path.isfile(os.path.join(BP.RUNS, run, s, ".done"))    # the trainer's flag: a model.pt alone may be unchecked (a failed export)
    if stage == "stage0":
        if not enabled(cfg, "stage0"):
            raise SystemExit("stage0.enabled is 0: the run uses the stored B' (set it to 1 to train one)")
        from trainkit import control
        from lane_center.training.train import qgraph_train
        step_dir = os.path.join(run_dir, "stage0")
        os.makedirs(step_dir, exist_ok=True)
        control.clear_stale(step_dir)                                                  # once, before the parts: a stop request during one of them is obeyed by the next
        for part in STAGE0:
            if not complete(cfg["name"], f"stage0/{part}"):
                qgraph_train.main(stage0_args(cfg, part), step_dir=step_dir)
        last = os.path.join(run_dir, "stage0", STAGE0[-1], "model.pt")
        shutil.copyfile(last, os.path.join(run_dir, "stage0", "model.pt.part"))
        os.replace(os.path.join(run_dir, "stage0", "model.pt.part"), os.path.join(run_dir, "stage0", "model.pt"))
        open(os.path.join(run_dir, "stage0", ".done"), "w").close()
        return 0
    if stage == "tokens":
        if not enabled(cfg, "tokens"):
            raise SystemExit("tokens.enabled is 0: stage 1 reads the stored token cache (set it to 1 for the run's own)")
        b = tokens_b(cfg)
        if enabled(cfg, "stage0") and not complete(cfg["name"], "stage0"):
            raise SystemExit(f"the run's stage 0 is not complete: no B' for its tokens ({b})")
        from lane_center.net.model import jtok as JT
        from lane_center.net.model import lane_net as LN
        from lane_center.net.prep import lane_tokcache as LC
        t = cfg["tokens"]
        for split in ("train", "val"):                                                 # TRAIN and VAL only: the holdout rows never
            a = argparse.Namespace(run=b, split=split, K=int(t["K"]), f2=bool(t["f2"]), limit=0, workers=int(t["workers"]), out=own_tokens(cfg))
            LC.build(a, lambda run, dev: LN.LaneNet(LN.load_b(run, dev), K=a.K), JT.pair_geometry)
        return 0
    if stage == "stage1":
        src = source(cfg)
        if complete(src, "stage1"):
            print(f"stage 1 of {src} is complete")
            return 0
        if src != cfg["name"]:
            raise SystemExit(f"from_run {src}: its stage 1 is not complete")
        from lane_center.training.train import jtoken_train
        return jtoken_train.main(stage_args(cfg, "stage1")) or 0
    if stage == "stage2":
        if not complete(cfg["name"], "stage2"):
            from lane_center.training.train import jlive_train
            code = jlive_train.main(stage_args(cfg, "stage2")) or 0
            if code:
                return code
        shutil.copyfile(os.path.join(run_dir, "stage2", "model.pt"), os.path.join(run_dir, "model.pt.part"))
        os.replace(os.path.join(run_dir, "model.pt.part"), os.path.join(run_dir, "model.pt"))
        return 0
    raise SystemExit(f"stage {stage!r}: stage0, tokens, stage1 or stage2")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
