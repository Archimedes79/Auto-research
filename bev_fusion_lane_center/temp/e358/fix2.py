import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
# docstring
("""    wsl bash tools/env/_py14.sh jwire_train.py cv --seed 0 [--pair-run e312_p_s0] [--folds 5] [--epochs 60]   # DIAGNOSTIC""", """    wsl bash tools/env/_py14.sh jwire_train.py cv --seed 0 [--pair-run e312_p_s0] [--ends kp|att] [--folds 5] [--epochs 60]   # DIAGNOSTIC"""),
("""    wsl bash tools/env/_py14.sh jwire_train.py run --seed 0 [--name e325_w_s0 | --pair-run e324_s_s0 --name e350_w_s0] [--heads 3] [--epochs 60] [--max-train N]   # the head of the card: through run.py (E325.w_s<S>): TRAIN records (a live pass, GPU) -> the pilot, the jitter ->
                                                                                                 # the heads (3 head seeds + the distance-only ablation) -> `runs/<name>/model.pt` (head seed 0 = the arm), `model_h<k>.pt`, `wire_agreement.json`""", """    wsl bash tools/env/_py14.sh jwire_train.py run --seed 0 [--name e325_w_s0 | --pair-run e324_s_s0 [--ends att] --name e358_w_s0] [--heads 3] [--epochs 60] [--max-train N]   # the head of the card: through run.py (E325.w_s<S>): TRAIN records (a live pass, GPU) -> the pilot, the jitter ->
                                                                                                 # the heads (3 head seeds + the distance-only ablation + the no-jitter head `wn`) -> `runs/<name>/model.pt` (head seed 0 = the arm), `model_h<k>.pt`, `model_wn.pt`, `wire_agreement.json`"""),
("""`model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard; the default name e325_w_s<seed> is E325's run); with `--pair-run` the run needs its own `--name`; a name is a folder name.
""", """`model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard; the default name e325_w_s<seed> is E325's run); with `--pair-run` the run needs its own `--name`; a name is a folder name.
The END KIND (E358, `--ends kp|att`, default kp): the records hold the keypoints (E325, E350) or the ATTACHED ends of the pair run (`jwire_data`: att needs `--pair-run` of an attach model), it reaches the VAL records, the live pass, the cache
names and metas, and the head's checkpoint args and `wire_agreement.json` (`ends`). The jitter is calibrated on the error of the pair ends of the records (the attached ends under att). The no-jitter head `wn` (E358, reported only) is the
same training on the records without any perturbation of the pair ends: `model_wn.pt` and `wire_agreement.json['no_jitter_head']`; it never replaces a head file.
"""),
# run_target
("""def run_target(seed, name, pair_run=None):
    \"\"\"-> (run name, pair run) of a training run. Default (pair_run None): E312's records and the name e325_w_s<seed> (E325); with a pair run `name` is mandatory. A name is a folder name, and a run folder that already
    holds a `wire_agreement.json` or a `model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard: a second call with the default name would overwrite E325's run)\"\"\"
    given = pair_run is not None
    pair_run = JD.resolve_pair_run(seed, pair_run)""", """def run_target(seed, name, pair_run=None, ends="kp"):
    \"\"\"-> (run name, pair run) of a training run. Default (pair_run None): E312's records and the name e325_w_s<seed> (E325); with a pair run `name` is mandatory. A name is a folder name, and a run folder that already
    holds a `wire_agreement.json` or a `model.pt` is refused in EVERY case (`trainer.Run` has no overwrite guard: a second call with the default name would overwrite E325's run); ends att needs a pair run with attached ends\"\"\"
    given = pair_run is not None
    pair_run = JD.resolve_pair_run(seed, pair_run)
    JD.check_ends(ends, seed, pair_run)"""),
])
print("ok")
