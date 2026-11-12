import re
p = "evaluation/jrule_removal.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''READOUTS = (("unsnapped (differentiable)", "ends_card_thr0.1.json"), ("snapped (rule step)", "ends_card_snap_thr0.1.json"))''',
'''READOUTS = {"plain": ("unsnapped (differentiable)", "ends_card_thr0.1.json"), "snap": ("snapped (rule step)", "ends_card_snap_thr0.1.json"), "attach": ("attached (E320, differentiable)", "ends_card_attach_thr0.1.json")}''')
s = s.replace('''def run(arm, control, seeds, max_loss, root=None):
    verdict, missing = True, False
    for label, name in READOUTS:''', '''def run(arm, control, seeds, max_loss, readouts=("plain", "snap")):
    verdict, missing = True, False
    for label, name in (READOUTS[r] for r in readouts):''')
s = s.replace('''    ap.add_argument("--max-loss", type=float, default=0.01)
    a = ap.parse_args()
    sys.exit(run(a.arm, a.control, a.seeds, a.max_loss))''', '''    ap.add_argument("--max-loss", type=float, default=0.01)
    ap.add_argument("--readouts", nargs="+", default=["plain", "snap"], choices=sorted(READOUTS), help="plain = unsnapped, snap = snapped; attach = the E320 attached read-out (re-based arms: --readouts attach)")
    a = ap.parse_args()
    sys.exit(run(a.arm, a.control, a.seeds, a.max_loss, a.readouts))''')
s = s.replace('''Read-outs (D85: the unsnapped one is the differentiable one and counts; the snapped one, the rule step of E311, is reported and must hold too): `ends_card_thr0.1.json` (connectors at p >= 0.1 drawn from the keypoints) and
`ends_card_snap_thr0.1.json` (the same, ends snapped to the delivered pieces), both written by `jends_eval.py score` into `runs/<run>/` (`--thr 0.1` with and without `--snap`).''', '''Read-outs (D85: the unsnapped one is the differentiable one and counts; the snapped one, the rule step of E311, is reported and must hold too): `plain` = `ends_card_thr0.1.json` (connectors at p >= 0.1 drawn from the keypoints) and
`snap` = `ends_card_snap_thr0.1.json` (the same, ends snapped to the delivered pieces), both written by `jends_eval.py score` into `runs/<run>/` (`--thr 0.1` with and without `--snap`); `attach` = `ends_card_attach_thr0.1.json` (E320: connector ends attached by the network, no snap) once E320 is the reference.''')
open(p, "w", encoding="utf-8").write(s)
for f, old, new in (("temp/e332/chain.sh", "--control $CONTROL_PREFIX >>", "--control $CONTROL_PREFIX --readouts $RULE_READOUTS >>"), ("temp/e333/chain.sh", "--control $CONTROL_PREFIX >>", "--control $CONTROL_PREFIX --readouts $RULE_READOUTS >>")):
    t = open(f, encoding="utf-8").read()
    assert old in t
    open(f, "w", encoding="utf-8").write(t.replace(old, new))
t = open("temp/e330/recipe.sh", encoding="utf-8").read()
t += 'RULE_READOUTS="plain snap"                                             # the read-outs jrule_removal.py judges on: plain = unsnapped (differentiable), snap = the rule step; re-based on E320: "attach"\nREADOUT_EXTRA=""                                                         # re-based on E320: "--attach" (post_pair then also scores the attached read-out)\n'
t = t.replace("#   chain_lib.sh: the read-out flags of post_pair (E320's read-out replaces --snap; the unsnapped read-out stays as the second number)\n", "#   RULE_READOUTS=\"attach\" and READOUT_EXTRA=\"--attach\" (post_pair then scores E320's attached read-out next to the unsnapped and the snapped one; the rule judges on the attached one)\n")
open("temp/e330/recipe.sh", "w", encoding="utf-8").write(t)
t = open("temp/e330/chain_lib.sh", encoding="utf-8").read()
old = "  bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S >> $LOGD/score_$R.log 2>&1\n"
new = "  [ -n \"$READOUT_EXTRA\" ] && bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S $READOUT_EXTRA --thr 0.1 >> $LOGD/score_extra_$R.log 2>&1\n" + old
assert old in t
open("temp/e330/chain_lib.sh", "w", encoding="utf-8").write(t.replace(old, new))
print("ok")
