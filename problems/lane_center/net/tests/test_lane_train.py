"""Spec test of the entry of the two training stages (`training/train/lane_train.py`) and of the base config (`training/configs/base.json`):
every setting of a trainer is a key of the base config, so an arm of a card can set it (`research check` refuses a key the base config lacks).

    python -P -m lane_center.net.tests.test_lane_train          # CPU, seconds after the imports; part 3 reads the run root, nothing is written there

1. argv: `options` on a toy block (known answer); the argv of the base config parses to what the two argv lists of before this rule parsed to
   (OLD: the champion's recipe did not move); every key of a block, set to another value, reaches the trainer's option of its name; `from_run`.
2. keys: the keys of the blocks stage1 / stage2 and of every part of stage0 are exactly the options of their parsers, without what the run gives;
   the parts of stage0 are the recipes of B' (E271 a, E273 p, E292 a) chained through the run's folders; tokens.enabled points stage 1 at the run's cache.
3. recipe: the trainers take the argv of the base config for the recipe of the trained run REF (`check_start`: the same recipe is resumed) and
   refuse it with one setting changed. Skipped with a printed line where the run root does not hold REF.
An adoption that changes the champion's recipe changes OLD and REF here with it."""
import json
import os
import tempfile

from lane_center.training.train import lane_train as LT
from lane_center.training import bflc_paths as BP
from lane_center.training.train import jlive_train as JLT
from lane_center.training.train import jtoken_train as JKT
from lane_center.training.train import qgraph_train as QGT

BASE = json.load(open(os.path.join(BP.ROOT, "training", "configs", "base.json"), encoding="utf-8"))
PARSE = {"stage1": JKT.parse_args, "stage2": JLT.parse_args}
REF = "E398.retrain_s0"                                                    # the champion's recipe trained from the base config (2026-11-27)
# the argv of the run E1.a_s1 (seed 1) as lane_train built them before it passed the whole blocks
OLD = {"stage1": ["--name", "E1.a_s1/stage1", "--seed", "1", "--head-seed", "3", "--ident", "e292_a", "--none-weight", "0.25", "--steps", "12000",
                  "--bs", "8", "--lr", "0.001", "--workers", "4"],
       "stage2": ["--name", "E1.a_s1/stage2", "--seed", "1", "--head-run", "E1.a_s1/stage1", "--train-nodes", "1", "--live-inputs", "1",
                  "--steps", "3000", "--bs", "8", "--lr", "0.0001", "--workers", "4"]}
# another value for every key of a block (stage 2 in two sets: the pyramid needs train_nodes and live_inputs at 1); a new key gets a value here
SET = {"stage1": [dict(ident="e273_p", head_seed_offset=5, none_weight=0.5, steps=7, bs=3, lr=0.5, workers=1)],
       "stage2": [dict(head_run="h_s{seed}", b_run="runs/b_s{seed}", b_state_from="c_s{seed}", head_seed=9, b_weight=0.5, train_pyr=1, pyr_lr_mult=0.1, steps=7, bs=3,
                       lr=0.5, workers=1),
                  dict(train_nodes=0, live_inputs=0, output_layers="soft_output")]}
RUN_GIVES = ("name", "seed", "smoke")                                      # no setting of a card: the run's name and seed; the 20-step plumbing run
STAGE0_GIVES = ("head", "resume_from")                                     # the chain of stage 0: the run's own E271 head and the part before


def cfg_of(name="E1.a_s1", seed=1, **top):
    return dict(BASE, name=name, seed=seed, out_root=BP.RUNS, **top)


def with_block(cfg, stage, **settings):
    return dict(cfg, **{stage: dict(cfg[stage], **settings)})


def parsed(cfg, stage):
    return vars(PARSE[stage](LT.stage_args(cfg, stage)))


def refused(fn, text):
    try:
        fn()
    except (SystemExit, AssertionError) as e:
        assert text in str(e), f"refused, but not for {text!r}: {e}"
        return
    raise RuntimeError(f"accepted although it must be refused ({text})")


def test_argv():
    assert LT.options({"none_weight": 0.25, "bs": 8, "lr": 1e-4, "head_run": None, "b_run": "runs/e292_a_s{seed}"}, 1) == \
        ["--none-weight", "0.25", "--bs", "8", "--lr", "0.0001", "--b-run", "runs/e292_a_s1"]
    refused(lambda: LT.options({"smoke": True}, 0), "no value of an option")                 # a flag without a value has no key form
    cfg = cfg_of()
    for stage in PARSE:
        assert parsed(cfg, stage) == vars(PARSE[stage](OLD[stage])), f"{stage}: the base config is no longer the recipe of before"
        assert {k for s in SET[stage] for k in s} == set(BASE[stage]), f"{stage}: SET names other keys than the block {sorted(BASE[stage])}"
        for other in SET[stage]:
            assert all(BASE[stage][k] != v for k, v in other.items()), f"{stage}: a value of SET is the base config's"
            got = parsed(with_block(cfg, stage, **other), stage)
            want = {k: v.format(seed=1) if isinstance(v, str) else v for k, v in other.items()}
            if "head_seed_offset" in want:
                want["head_seed"] = 1 + want.pop("head_seed_offset")
            assert {k: got[k] for k in want} == want, f"{stage}: {want} set in the block, the trainer reads {({k: got[k] for k in want})}"


def test_from_run():
    """`from_run`: stage 2 starts from that run's stage 1, the step stage1 trains nothing; a folder outside the run root and a second name
    for the head are refused"""
    for form in (os.path.join(BP.RUNS, "E9.b_s1"), "runs/E9.b_s1"):                          # as the worker writes a card's `needs`; as a record names a run
        got = parsed(cfg_of(from_run=form), "stage2")
        assert got["head_run"] == "E9.b_s1/stage1" and got["name"] == "E1.a_s1/stage2", got
    refused(lambda: LT.stage_args(cfg_of(from_run="/elsewhere/E9.b_s1"), "stage2"), "not a run below the run root")
    refused(lambda: LT.stage_args(with_block(cfg_of(from_run="runs/E9.b_s1"), "stage2", head_run="h"), "stage2"), "both name the head of stage 1")
    trained = []
    keep = BP.RUNS, JKT.main
    with tempfile.TemporaryDirectory() as tmp:
        try:
            BP.RUNS, JKT.main = tmp, lambda argv: trained.append(argv)                       # (the run root of this test; no real training)
            for name, top in (("own", {}), ("other", dict(from_run=os.path.join(tmp, "E9.b_s1")))):
                with open(os.path.join(tmp, name + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(cfg_of(**top), fh)
            refused(lambda: LT.main(["stage1", os.path.join(tmp, "other.json")]), "its stage 1 is not complete")
            os.makedirs(os.path.join(tmp, "E9.b_s1", "stage1"))
            open(os.path.join(tmp, "E9.b_s1", "stage1", ".done"), "w").close()
            assert LT.main(["stage1", os.path.join(tmp, "other.json")]) == 0 and not trained, "stage 1 was trained although from_run has one"
            assert LT.main(["stage1", os.path.join(tmp, "own.json")]) == 0 and trained == [LT.stage_args(cfg_of(), "stage1")], trained
        finally:
            BP.RUNS, JKT.main = keep


def test_keys():
    need = {"stage1": ["--name", "x"], "stage2": ["--name", "x", "--head-run", "h", "--train-nodes", "1"]}
    for stage, parse in PARSE.items():
        keys = {"head_seed" if k == "head_seed_offset" else k for k in BASE[stage]}          # stage 1's head seed is seed + the offset
        want = set(vars(parse(need[stage]))) - set(RUN_GIVES)
        assert keys == want, (f"{stage}: options of the trainer without a key of base.json {sorted(want - keys)}, keys without an option "
                              f"{sorted(keys - want)}")
    want = set(vars(QGT.parse_args(["--name", "x"]))) - set(RUN_GIVES) - set(STAGE0_GIVES) - {"soft_nms"}       # soft_nms: made from soft_nms_mnms
    assert set(BASE["stage0"]) == {"enabled"} | set(LT.STAGE0), sorted(BASE["stage0"])
    for part in LT.STAGE0:
        assert set(BASE["stage0"][part]) == want, f"stage0.{part}: options of qgraph_train without a key {sorted(want - set(BASE['stage0'][part]))}, keys without one {sorted(set(BASE['stage0'][part]) - want)}"


def test_stage0_tokens():
    """the parts of stage 0 from the base config are the recipes of the champion's B' (E271 a, E273 p, E292 a: v1 ledger), chained through the run's folders;
    tokens.enabled makes stage 1 read the run's own cache"""
    cfg = dict(cfg_of(), stage0=dict(BASE["stage0"], enabled=1), tokens=dict(BASE["tokens"], enabled=1))
    got = {p: vars(QGT.parse_args(LT.stage0_args(cfg, p))) for p in LT.STAGE0}
    assert {k: got["e271"][k] for k in ("phases", "steps", "bs", "width", "gate_phases")} == dict(phases=[1], steps=[4000], bs=8, width=48, gate_phases=[]), got["e271"]
    assert got["e273"]["steps"] == [QGT.STEPS[k] for k in range(2, 8)] and got["e273"]["gate_phases"] == [] and got["e273"]["soft_nms"] is None and got["e273"]["resume_from"] is None
    assert got["e273"]["head"] == got["e292"]["head"] == "runs/E1.a_s1/stage0/e271/model.pt" and got["e292"]["resume_from"] == "runs/E1.a_s1/stage0/e273/model.pt"
    assert {k: got["e292"][k] for k in ("phases", "steps", "bs", "gate_phases", "gate_node_prec", "gate_node_rec", "soft_nms", "name", "seed")} == dict(
        phases=[3, 4, 5, 6, 9], steps=[3000, 2500, 1000, 750, 1000], bs=4, gate_phases=[3], gate_node_prec=0.75, gate_node_rec=0.5, soft_nms=["mnms", 1.75, 0.1],
        name="E1.a_s1/stage0/e292", seed=1)
    assert parsed(cfg, "stage1")["ident"] == "runs/E1.a_s1/tokens" and LT.tokens_b(cfg) == "runs/E1.a_s1/stage0"
    assert LT.tokens_b(dict(cfg, stage0=BASE["stage0"])) == "runs/e292_a_s1", "tokens without stage 0: the stored B' of stage 1's ident"
    for stage in ("stage0", "tokens"):                                                         # the base config: both off, the steps refuse
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "c.json")
            json.dump(cfg_of(), open(p, "w", encoding="utf-8"))
            refused(lambda: LT.main([stage, p]), "enabled is 0")


def test_recipe():
    run = os.path.join(BP.RUNS, REF)
    if not os.path.isdir(run):
        print(f"part 3 skipped: {run} is not on this machine (the recipe of the base config is not compared with a trained run)")
        return False
    recipe = {"stage1": lambda c: parsed(c, "stage1"),       # jtoken_train.train stores these as the run's args (it changes only a smoke run's steps)
              "stage2": lambda c: JLT.resolve(JLT.parse_args(LT.stage_args(c, "stage2")), BP.RUNS)}
    check = {"stage1": JKT.check_start, "stage2": JLT.check_start}
    cfg = cfg_of(REF, 0)
    for stage, change in (("stage1", dict(none_weight=1.0)), ("stage2", dict(b_weight=0.0))):
        check[stage](recipe[stage](cfg), os.path.join(run, stage))
        refused(lambda: check[stage](recipe[stage](with_block(cfg, stage, **change)), os.path.join(run, stage)), "of another run")
    return True


if __name__ == "__main__":
    test_argv()
    test_from_run()
    test_keys()
    test_stage0_tokens()
    print("test_lane_train ok: the argv of the base config, every key of a block reaches its option, from_run; keys = options; stage 0 = the recipes of B', tokens; "
          + ("the recipe of " + REF if test_recipe() else "NO recipe check"))
