def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, s.count(a), a[:80])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)


edit("training/README.md", [("`potential` (step 0 of the E368 card: what a past could give, JSON), `selftest`", "`loader-fixture` (the token test again through the TRAINING path: the loader's items and `jpast.make_past`, the pose entry against a fresh `rel_pose` of the pair; the identity pose, the previous item's entry and a shifted previous frame must FAIL), `potential` (step 0 of the E368 card: what a past could give, JSON), `selftest`")])
edit("training/grid/README.md", [
    ("the roles of the weak tokens on a hand-made frame), `check` and the fixture on 40 real VAL pairs, then the textual mutants: every mutant must make the selftest fail",
     "the roles of the weak tokens on a hand-made frame, the TEST guard of `build` on a toy VAL cache), `check`, the fixture on 40 real VAL pairs, the loader fixture and the pinned numbers of the fixture (20 pairs) and of `potential` (60 frames), then the textual mutants: every mutant must make ITS command fail (the selftest, or the fixture / potential / loader fixture named in the mutant)"),
    ("one batch of the real TRAIN chains through the loop (7990 chains, same seed = same order); then the textual mutants",
     "one batch of the real TRAIN chains through the loop (7990 chains, same seed = same order) and `train()` with the real loader (the arguments it hands to `lane_past_data.loader`); the key-weight switch at counters 999 / 1000 (head, `make_past`, read-out), a change of every recipe key refused at a resume, `U_WINDOW` = the card's last 500 steps, the sibling run folders of the mode dumps read by `jtoken_eval.read_dump`, the carried state's RMS per position; then the textual mutants"),
])
edit("evaluation/README.md", [("`jpast_eval.py` | E368 S2 read-out: `dump` runs the token head with (arm r) or without (arm z) the past block frame by frame in the E365 cache order with the state carried (`--mode stream` = `token_<split>.pkl`, the file `jtoken_eval.py score` / `sweep` read; `truncated` (reset every `--window` frames), `off`, `shuffled` (the state of another scene at the same step index), `still` (identity pose) -> `token_<split>_<mode>.pkl`;",
                               "`jpast_eval.py` | E368 S2 read-out: `dump` runs the token head with (arm r) or without (arm z) the past block frame by frame in the E365 cache order with the state carried (`--mode stream` -> `runs/R/token_<split>.pkl`, the file `jtoken_eval.py score` / `sweep` read; `truncated` (reset every `--window` frames), `off`, `shuffled` (the state of another scene at the same step index), `still` (identity pose) -> the SIBLING run folder `runs/R_<mode>/` (`token_<split>.pkl` + a copy of model.pt), so that `jtoken_eval.py score --run R_<mode>` and `sweep` read it as a run;"),
                              ("`use` = u per stream position and what the past changed (stream against off) -> `runs/R/past_use_<split>.json`.", "`use` = u and the RMS of the carried state per stream position, the first 5 frames of every scene against the later ones (frame indices for the topo-F1 split), what the past changed (stream against off) -> `runs/R/past_use_<split>.json`.")])
print("ok")
