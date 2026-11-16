"""E341 second critic pass: card text (D94, tagged E312 rescoring, cost, ms timing), chain exit codes and host gate, README / docstring, a paint-channel test."""
import re


def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:90]
    return s.replace(a, b)


def wr(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)


p = "experiments/E341_evidence-without-paint-branch/experiment.md"
s = rd(p)
s = sub(s, "Against E312 only the snapped read-out has a counts file (`runs/e312_p_s<S>/ends_card_snap_thr0.1.counts.pkl`); for the unsnapped one, which counts under D85, E312 has none (re-scoring it with counts would rewrite a stored run record: not done), so the verdict against E312 on the unsnapped read-out is the point rule only, and the card says so.",
        "Against E312 the snapped read-out has a counts file (`runs/e312_p_s<S>/ends_card_snap_thr0.1.counts.pkl`); for the unsnapped one, which counts under D85, E312 has none, so the chain first re-scores E312 unsnapped with a TAGGED card (`jends_eval.py score --run e312_p_s<S> --seed <S> --thr 0.1 --tag cnt`: a new file `ends_card_cnt_thr0.1.json` with its counts, a stored card is never overwritten) and then runs `jdiag.py boot e341_p_s<S> ends_card_thr0.1 e312_p_s<S> ends_card_cnt_thr0.1`.")
s = sub(s, "about 1.9 h wall for both seeds in the two slots, 3.2 slot hours.", "about 3.0 h wall for both seeds one after the other (D98: by day one GPU job; E332's pair runs took 70 - 74 minutes each, the post-processing 0.25 h per seed), 3.2 slot hours.")
s = sub(s, "**Intervals (the rule stays the point estimate, D85):**", "**Control and D94:** the control stays E312 (the rule was written for it before the run); D94 makes E324 s the system of record, which is about 0.015 higher snapped, so ADOPT here means \"adopted on the E312 recipe\"; a transfer to the E324 s recipe needs its own run (a card after an ADOPT here), and this result is also read against E324 s afterwards.\n**Intervals (the rule stays the point estimate, D85):**")
s = sub(s, "the E341 checkpoint is timed in part B of E342.", "the E341 checkpoint (maps head and whole forward) is timed in part B of E342 (section 2 there names it).")
wr(p, s)

p = "experiments/E342_pair-head-host-time/experiment.md"
s = rd(p)
s = sub(s, "**GPU start rule:**", "Part B also times the maps head of the E341 checkpoint (`runs/e341_p_s0/model.pt`, no 0.5 m branch) against E312's (`runs/e312_p_s0/model.pt`), in the same process, blocks A B | B A: the ms that E341 saves, which E332 and E341 only estimate.\n**GPU start rule:**")
wr(p, s)

p = "training/grid/README.md"
s = rd(p)
s = sub(s, "`jmaps_train.py --no-paint --no-zone`; the pair head must then run `--no-evidence`.", "`jmaps_train.py --no-paint --no-zone`; the pair head then runs with the four-input evidence (E341, `JEnds(paint_ev=False)`, set by `JEndModel` from the ctx) or with `--no-evidence` (E332); `nozone` needs `--no-evidence`.")
wr(p, s)

p = "training/grid/jends.py"
s = rd(p)
s = sub(s, "    JEnds(d=64, order=True, evidence=True)(jout, P, gt=None) -> dict ", "    JEnds(d=64, order=True, evidence=True, paint_ev=True)(jout, P, gt=None) -> dict ")
s = sub(s, "    # E308: with node tokens the pair logits depend on the tokens (and the kp layer takes them)\n", """    jp = dict(jout)                                                                                    # E341: the two paint columns (4, 5) read their own channel: stop line about sigmoid(4) = 0.98, crossing about sigmoid(-4) = 0.02 (a swap of the two is caught)
    jp["j05"] = torch.cat([torch.full((B, 1, 200, 200), 4.0), torch.full((B, 1, 200, 200), -4.0)], 1)
    torch.manual_seed(0)
    mp = JEnds(d=d, order=True, evidence=True, paint_ev=True)
    mp.pair.register_forward_pre_hook(lambda _m, a_: seen.__setitem__("p", a_[0].detach().clone()))
    mp(jp, P, gt=gt)
    pc_ = seen["p"][..., c0 + 4: c0 + 6]
    assert abs(float(pc_[..., 0].mean()) - 0.982) < 0.01 and abs(float(pc_[..., 1].mean()) - 0.018) < 0.01, (float(pc_[..., 0].mean()), float(pc_[..., 1].mean()))
    # E308: with node tokens the pair logits depend on the tokens (and the kp layer takes them)
""")
wr(p, s)

p = "temp/e341/chain.sh"
wr(p, """#!/bin/bash
# E341 chain (D98): the pair stage on the E332 arm-a maps, the two seeds ONE AFTER THE OTHER (by day one GPU job, about 3.0 h wall), the post-processing, the rule and the intervals. Called by temp/day18/day18.sh.
# Exit code 1 when a pair run failed or left no model (a jrule exit 1 is a verdict, not a failure); marker temp/e341/E341_DONE only on exit 0. Never edit this script while it runs.
cd "$(dirname "$0")/../.."
mkdir -p temp/e341
L=temp/e341/chain.log; LOGD=temp/e341
. temp/e330/recipe.sh
. temp/e330/chain_lib.sh
host_ok() { [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 8 ]; }     # D98: free RAM >= 8 GB, load < 8
RC=0
for S in 0 1; do
  until can_start && host_ok; do echo "e341 seed $S waits for the window / the host $(date)" >> $L; sleep 300; done
  bash temp/e341/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; X=$?; echo "e341 seed $S exit $X $(date)" >> $L
  { [ $X = 0 ] && [ -f runs/e341_p_s$S/model.pt ]; } || { RC=1; echo "e341 seed $S FAILED (exit $X or no model) $(date)" >> $L; }
  post_pair e341_p_s$S $S
done
for S in 0 1; do                                                                                       # E312 unsnapped WITH counts: a tagged card, nothing stored is overwritten
  bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --tag cnt >> $LOGD/rescore_e312_s$S.log 2>&1
done
bash tools/env/_py14.sh jrule_removal.py --arm e341_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_e341.log 2>&1; echo "e341 rule exit $? (1 = not adopted, a verdict) $(date)" >> $L
for S in 0 1; do                                                                                       # intervals: against E332 arm a on both read-outs, against E312 on both
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e332_pa_s$S ends_card_thr0.1 >> $LOGD/boot_vs_a_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e332_pa_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_a_snap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 >> $LOGD/boot_vs_e312_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
done
[ $RC = 0 ] && touch temp/e341/E341_DONE
echo "done RC=$RC $(date)" >> $L
exit $RC
""")
print("ok")
