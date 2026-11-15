def sub(p, a, b, new=".new_s292", nth=1):
    s = open(p, encoding="utf-8").read()
    assert s.count(a) == 1, (p, a[:60])
    open(p + new, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))

# run_gates
sub("tools/maintenance/run_gates.sh", 'run "head: wiring twin of the nearest-piece rule, kept ends (E325)"  grid/jwire.py\n',
    'run "head: wiring twin of the nearest-piece rule, kept ends (E325)"  grid/jwire.py\nrun "head: wiring head Wire, features, labels, places, leak tests (E325)"  grid/jwire_head.py\nrun "head: wiring head data, records, agreement (E325)"  jwire_data.py selftest\nrun "head: wiring head training, ablation (E325)"  jwire_train.py selftest\n')

# training/grid README: row after jwire.py
p = "training/grid/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("| `jwire.py` |"))
L[i + 1:i + 1] = ["| `jwire_head.py` | E325 (step 2): the learned wiring head `Wire` (15 features per candidate PLACE, MLP 15 -> 32 -> 1 = 545 parameters, softmax over the places of an end), `piece_attrs` / `end_instance` / `build_instances` (features from the stored graph and the keypoint pair only, NEVER from the GT; the oracle label = the place of the piece the rule picks at the matched GT end, an end whose oracle place is not among the keypoint's places has no label), `pack`, `wire_loss` (labelled ends only), `wire_choice_head` (-> `add_connectors(wire_choice=)`), `Wire.rule_like()` (hand-set weights = the nearest-piece rule: the fixture of the read-out path), `load_wire` | new | `python jwire_head.py` (twin = rule-like head on 516 random pairs, leak test: a feature never reads the GT, the oracle label differs from the keypoint rule where it should, side-correct node / link attributes, float64 distances, 545 parameters, gradient on labelled ends only, a synthetic learning check) |"]
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write("\n".join(L))

# evaluation README: row after jwiring.py
p = "evaluation/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("| `jwiring.py` |"))
L[i + 1:i + 1] = ["| `jwire_data.py` | E325 (step 2): the data of the wiring head: per frame a compact RECORD (kept pieces and attributes, pairs, keypoint probabilities, matched GT ends), `val` (VAL records from the stored dumps + the rule's agreement with the oracle label: 85.6 / 85.4 % overall, 75.7 / 75.1 % on the real choices), `live` (the same records from a LIVE pass of B and the pair run on any split, GPU: 'train' for the head, 'val_clean' reproduces the stored-dump records), `pilot_stats`, `make_jitter` (heavy-tailed keypoint jitter), `calibrate_jitter` (matched on the median keypoint error) | `python jwire_data.py selftest`; `python jwire_data.py val --seed 0` |"]
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write("\n".join(L))

# training README: row after run_config
p = "training/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("| `train/run_config.py` |"))
L[i + 1:i + 1] = ["| `train/jwire_train.py` | E325 (step 2): training and diagnostics of the wiring head `Wire`: `cv` (scene-wise cross-validation on the VAL records, IN-SAMPLE diagnostic, with the distance-only ablation), `run` (the card's training run through `run.py`: TRAIN records by a live pass, pilot, jitter calibration, 3 head seeds + the ablation, `runs/<name>/model.pt`, `wire_agreement.json`) | `python jwire_train.py selftest` |"]
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("ok")
