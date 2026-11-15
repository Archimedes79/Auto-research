import re
# grid README: rows after jwire_head.py
p = "training/grid/README.md"
s = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(s) if l.startswith("| `jwire_head.py`"))
rows = [
 "| `jnarrow.py` | E353 (D99 step 1): the junction maps head narrowed PHYSICALLY by channel surgery on a trained `JEndModel` (spec = plain JSON `{calib, edits: [{unit | tap, frac}]}`, stored in the checkpoint as `args['narrow']`): whole GroupNorm groups are removed, the lowest summed E345 importance (`calib.pt`), so the kept channels keep their statistics; the producer conv, its GroupNorm and every consumer conv are sliced, the removed channels' calibration means (post-ReLU) fold into the consumer's bias (a NEW bias tensor when it had none). Exact inside the map; the zero-padding border ring differs by the folded-bias term (documented, measured, tested). `resolve`, `narrow`, `build_narrowed`, `load_model` (`jends_eval.load_model`'s contract for a narrowed checkpoint), `expected_saving` (`jprune_cost.Cost`), `measured_macs` (FlopCounterMode), `difference` (against the hook-ablated model). Built: maps-head hidden tensors only; the pyramid, the residual stream, head outputs and the pair head are refused (E352 / not worth it). No digested file is edited | new (imports `jprune_probe`, `jprune_cost`, `jmodel_cost`) | `python test_jnarrow.py` |",
 "| `test_jnarrow.py` | E353: the groups against a numpy reference, the refusals, the narrowed model against the hook-ablated one (interior float tolerance, the border term by hand, heat / keypoints identical), MACs and parameters against `Cost` and FlopCounterMode, the spec round trip, the dump wrapper (unnarrowed dump == `jends_eval.dump`, `jends_eval.score` accepts the checkpoint), the trainer (frozen set, checks, records, resume, refusal, n1z, cp), the timing machinery on a narrowed toy; CPU, real E324 s and 8 VAL frames, about 7 minutes | new | itself |",
]
s[i + 1:i + 1] = rows
open(p, "w", encoding="utf-8").write("\n".join(s))
# training README: the train/ row
p = "training/README.md"
t = open(p, encoding="utf-8").read()
old = "`qnode_pilot.py` (E282: the CPU mini-pilot of the zone-matching node loss: cache of the frozen pyramid outputs, four arms on the shared trainer, the zone metrics and the gate of the card) |"
assert old in t
new = old[:-2] + ", `jnarrow_train.py` (E353: the e2e fine-tune of a stored pair run from its FULL state, `--narrow spec.json` = the maps head narrowed by `grid/jnarrow.py`, empty = the equal-budget control, `--steps 0` = the narrowed model before any training; `dump | score | time` subcommands = the stored evaluation tools through `jnarrow_eval.py`), `jnarrow_eval.py` (E353: `dump` / `score` / `time` with a surgery-aware model loader installed in its own process, so a narrowed checkpoint goes through the stored `jends_eval.py` and `jtiming.py` unchanged) |"
t = t.replace(old, new)
open(p, "w", encoding="utf-8").write(t)
# run_all_tests.sh: one line after the jprune probe test
p = "tools/maintenance/run_all_tests.sh"
t = open(p, encoding="utf-8").read()
k = 'test_jprune_probe.py\n'
assert k in t
t = t.replace(k, k + 'run "E353 narrowing of the junction maps head: groups, equality with the hook-ablated model, MACs = cost table, spec round trip, dump wrapper, trainer (frozen set, records, resume, n1z, cp), timing machinery (CPU, about 7 minutes)" grid/test_jnarrow.py\n', 1)
open(p, "w", encoding="utf-8", newline="\n").write(t)
