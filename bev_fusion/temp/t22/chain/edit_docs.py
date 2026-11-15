# scratch: README / architecture row edits of the stage-B chain step (plain string replaces, each must match exactly once)
import re
def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))

sub("training/README.md",
    "no camera-parameter gate in this version; not yet in `model.CAMERAS` (stage B) | `../tests/test_student_encoder.py` (shapes and softmax with five mutants, H8 geometry by closed forms, gradients, overfit, round trip) |",
    "no camera-parameter gate in this version. Stage B (in the chain, `ModelConfig.camera_student`, off by default): the same module is `model.student` (`model.STUDENTS`: `evit_b1`, `resnet18`; built last, so every other tensor's init is unchanged); the file also holds the distillation loss (`cell_maps`, `total_of`, `distill_terms`; `distill_student.py` and `train.forward_loss` use it), the anchor (`anchor_terms` of the model's `student_side`, `student_anchor_weight`: start, linear to start x floor at the fraction `until`, constant after), `keep_bn_eval` (BatchNorm statistics fixed under `.train()`) and `load_trained` / `newest_checkpoint` (a run folder or file -> the EMA weights, strictly) | `../tests/test_student_encoder.py` (shapes and softmax with five mutants, H8 geometry by closed forms, gradients, overfit, round trip), `../tests/test_student_chain.py` (the chain) |")
sub("training/README.md",
    "`--resume` guard (exit 2), `--eval-every`; `python training/distill_student.py --name A1_s0 --split small --freeze-backbone` |",
    "`--resume` guard (exit 2), `--eval-every`; `python training/distill_student.py --name A1_s0 --split small --freeze-backbone`; the loss and the checkpoint loader are in `student_encoder.py` |")
sub("training/architecture.md",
    "`LIFTS` pd / p / s; `COLLAPSES` zconv / zsum |\n| radar encoder |",
    "`LIFTS` pd / p / s; `COLLAPSES` zconv / zsum; option `camera_student` (t22 stage B, off): a small trained image encoder (`STUDENTS` evit_b1 / resnet18) replaces the cached teacher outputs of the CURRENT frame of both sources from `lane_img`, older frames keep the cached ones |\n| radar encoder |")
