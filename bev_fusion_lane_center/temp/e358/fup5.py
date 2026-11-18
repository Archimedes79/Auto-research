import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
apply(R + "training/train/jwire_train.py", [
    ("""same training on the records without any perturbation of the pair ends: `model_wn.pt` and `wire_agreement.json['no_jitter_head']`; it never replaces a head file.
""", """same training on the records without any perturbation of the pair ends (same epochs, learning rate, batch size and weight decay as head 0): `model_wn.pt` and `wire_agreement.json['no_jitter_head']`; it never replaces a head file.
The pilot (`pilot_train` / `pilot_val`) is measured on the unperturbed records, the heads train on the jittered draws: `wire_agreement.json['pilot_train_jittered']` = `near05` of the jittered draws (`near_share`, the share of the
oracle-labelled ends whose nearest piece end is within 0.5 m, from the features the head sees), the jitter scale and the number of draws; the verdict prints it next to the unperturbed TRAIN and the VAL value as a note.
"""),
])
apply(R + "training/README.md", [
    ("""`wn` = the no-jitter head (the same training on the unperturbed records, `model_wn.pt` and `wire_agreement.json['no_jitter_head']`, reported only);""", """`wn` = the no-jitter head (the same training on the unperturbed records, `model_wn.pt` and `wire_agreement.json['no_jitter_head']`, reported only); `wire_agreement.json['pilot_train_jittered']` = the `near05` share of the jittered draws the heads train on (`near_share`) next to the unperturbed pilot;"""),
    ("""`wn` trained on the unperturbed records and no head file replaced) |""", """`wn` trained on the unperturbed records with the run's epochs and no head file replaced, the jittered share known answer) |"""),
])
apply(R + "evaluation/README.md", [
    ("""the pilot's `near05` share (TRAIN against VAL, above 0.10 = shift-dominated) and the no-jitter head reported |""", """the pilot's `near05` share (unperturbed TRAIN against VAL, above 0.10 = shift-dominated; the jittered draw's share is a note), the no-jitter head reported (`--tag-wn`: its cards, topo F1 against c and w) and the prediction hit window of the card (`--prediction`, `--window`: per seed w - c, HIT / MISS) |"""),
    ("""the share flag); `python jwire_verdict.py`""", """the share flag, the jittered-share note, the no-jitter cards, the hit window); `python jwire_verdict.py`"""),
])
