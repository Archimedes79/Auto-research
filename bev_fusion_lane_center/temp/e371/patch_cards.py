import os


def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:60])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)


patch("experiments/E370_e365-rule-distilled-init/experiment.md", [
    ("stage B = the mean GT loss over its steps 1801 - 2000 not below the GT loss of the stage-A head at stage-B step 0 (measured before the first update and written here)",
     "stage B = the mean GT loss over its steps 1801 - 2000 not below the GT loss of the stage-A head at stage-B step 0 (the reference is the `gt_loss` of `jtoken_eval.py fixture --run e370_a_s<S> --split train --every 3`: eval mode, 291 frames, written here before stage B starts; the trainer's 'first loss of this process' is one batch and only reported)"),
])
patch("experiments/E369_e365-rule-prior-token-head/experiment.md", [
    ("the destination evidence b; the structural key bias",
     "the destination evidence b; the handle-scale head starts at c-rule's scale 1/3 (zero weight, the bias logit(0.2361); without it the step-0 topo F1 differed by -0.0022 on 16 TRAIN frames, with it by exactly 0); the structural key bias"),
])
patch("experiments/E371_e366-confirmation-fresh-head-seeds/experiment.md", [
    ("**Statement:** the E366 result is a property of the recipe",
     "**Records:** the run's config.json top-level 'seed' is the HEAD seed, `args.seed` and run.py's `--seed` (the ledger) are the CACHE seed: the ledger notes and the result say 'cache seed S, head seed S + 2'. The code critic of commit 81f9374b4b checked `--head-seed` (accept: the head initialisation, the batch order and the token-drop draws follow the head seed, the loader the cache seed; the live train.log of cf_s0 says 'cache seed 0, head seed 2'). **Statement:** the E366 result is a property of the recipe"),
])
print("ok")
