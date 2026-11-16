def rd(p):
    return open(p, encoding="utf-8").read()


def sub(s, a, b):
    assert s.count(a) == 1, a[:90]
    return s.replace(a, b)


p = "experiments/E341_evidence-without-paint-branch/experiment.md"
s = rd(p)
s = sub(s, "Card of 2026-11-14 17:00, written", "Card of 2026-11-14 16:00 (revised 16:50 after the critic), written")
s = sub(s, "(compute arm, D92 / D86)", "(compute arm, D92 / D86; the saving in MACs is large, the saving in ms per frame is an estimate of about 1.3 of 17.1 ms and is measured in part B of E342)")
s = sub(s, ' "kill": "the pair loss is not finite, or `runs/e332_ma_s<S>/model.pt` is missing (stage 1 itself passed its kill check in E332: maps keypoint F1 0.7409 / 0.7355 against E307 0.7233 / 0.7051)",',
        ' "kill": "the pair loss is not finite, or `runs/e332_ma_s<S>/model.pt` is missing, or the head the run builds has not 150 pair inputs (`pair.in_features`: 152 minus the two paint columns; the strict load of the dump stage and `grid/test_jmaps_simplify.py` check it); stage 1 itself passed its kill check in E332 (maps keypoint F1 0.7409 / 0.7355 against E307 0.7233 / 0.7051)",')
s = sub(s, "The reading of the four-input evidence is the paired difference to E332 arm a on the same seeds (`jdiag.py boot`): it says how much of the evidence loss the four inputs recover.",
        "**Intervals (the rule stays the point estimate, D85):** against E332 arm a (same maps, no evidence) on both read-outs: `jdiag.py boot e341_p_s<S> ends_card_thr0.1 e332_pa_s<S> ends_card_thr0.1` (unsnapped) and `... ends_card_snap_thr0.1 ... ends_card_snap_thr0.1` (snapped); the chain runs them. Against E312 only the snapped read-out has a counts file (`runs/e312_p_s<S>/ends_card_snap_thr0.1.counts.pkl`); for the unsnapped one, which counts under D85, E312 has none (re-scoring it with counts would rewrite a stored run record: not done), so the verdict against E312 on the unsnapped read-out is the point rule only, and the card says so.")
s = sub(s, "a loss near -0.012 would be NOT adopted). Cost:", "a loss near -0.012 would be NOT adopted). **Power:** the prediction is 0.002 inside the bound; with a paired spread of 0.004 - 0.006 per seed (E312's two seeds differ by 0.0044; the E331 / E332 seed effects differ by about 2 times; the critic's estimate, not measured) a result equal to the prediction passes both seeds on both read-outs only about 40 % of the time; a TIE or NOT adopted at -0.011 is inside the noise, and the paired difference to E332 arm a is the number that is worth reading. E331 / E332 also showed that the loss sits at the fixed operating point p >= 0.1 (pair precision at E312's recall was higher), so a calibration shift of the pair logits can decide the verdict, not the ranking. Cost:")
s = sub(s, "**What it saves when adopted:** the 0.5 m branch (`m05`, `head05`: 1.477 GMACs, 0.037 M parameters), its loss, the paint sidecars of the GT (`data/prep/junction_paint.py`) in the loader and the two paint inputs of the pair MLP; width, depth, resolution of everything else unchanged.",
        "**What it saves when adopted:** the 0.5 m branch (`m05`, `head05`: 1.477 GMACs, 0.037 M parameters), its loss and the two paint inputs of the pair MLP; width, depth and resolution of everything else unchanged. In ms per frame that is an estimate of about 1.3 ms of the 17.09 ms forward (E342: the maps head takes 2.82 ms for 3.256 GMACs; E332 did not measure ms): the E341 checkpoint is timed in part B of E342. NOT saved yet: the loader still builds the paint layer (`batch[\"jpaint\"]`, read unconditionally by `jmaps.py` and `jmaps_eval.py`) and the sidecars of `data/prep/junction_paint.py` stay needed; skipping them is a follow-up after an adoption (the E332 card says the same).")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "temp/e341/chain.sh"
s = rd(p)
s = sub(s, """for S in 0 1; do
  ( wait_start; bash temp/e341/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; echo "e341 seed $S exit $? $(date)" >> $L; post_pair e341_p_s$S $S ) &
done
wait
bash tools/env/_py14.sh jrule_removal.py --arm e341_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_e341.log 2>&1; echo "e341 rule exit $? $(date)" >> $L
""", """# D98: by day (weekends) ONE job at a time and only while the host has room; at night the second seed follows at once anyway
host_ok() { [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 8 ] && [ "$(awk '{print int($1)}' /proc/loadavg)" -lt 8 ]; }
for S in 0 1; do
  until host_ok; do sleep 60; done
  wait_start; bash temp/e341/start_pairs.sh $S >> $LOGD/lane$S.log 2>&1; echo "e341 seed $S exit $? $(date)" >> $L; post_pair e341_p_s$S $S
done
bash tools/env/_py14.sh jrule_removal.py --arm e341_p_s --control $CONTROL_PREFIX --readouts $RULE_READOUTS >> $LOGD/rule_e341.log 2>&1; echo "e341 rule exit $? $(date)" >> $L
for S in 0 1; do                                                                                       # intervals: against E332 arm a on both read-outs, against E312 on the snapped one (E312 has no unsnapped counts file)
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e332_pa_s$S ends_card_thr0.1 >> $LOGD/boot_vs_a_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e332_pa_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_a_snap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> $LOGD/boot_vs_e312_snap_s$S.log 2>&1
done
""")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "temp/e341/start_pairs.sh"
s = rd(p)
s = sub(s, "  -- bash tools/env/_pyr.sh jends_train.py", "  -- nice -n 10 bash tools/env/_pyr.sh jends_train.py")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "temp/day18/day18.sh"
s = rd(p)
s = sub(s, """bash temp/e341/chain.sh >> temp/day18/e341_chain.out 2>&1; echo "e341 chain exit $? $(date)" >> $L
touch temp/day18/DAY18_DONE
echo "done $(date)" >> $L""", """bash temp/e341/chain.sh >> temp/day18/e341_chain.out 2>&1; RC=$?; echo "e341 chain exit $RC $(date)" >> $L
[ $RC = 0 ] && touch temp/day18/DAY18_DONE                                                              # a failed chain leaves no done marker: a re-run needs only temp/e341/E341_DONE removed
echo "done $(date)" >> $L""")
open(p, "w", encoding="utf-8", newline="\n").write(s)

p = "temp/e330/chain_lib.sh"
s = rd(p)
a = 'can_start() { H=$(date +%-H); M=$(date +%-M); DOW=$(date +%u); [ "$H" -ge 22 ] ||'
assert s.count(a) == 1
s = s.replace(a, 'can_start() { H=$(date +%-H); M=$(date +%-M); DOW=$(date +%u); [ "$DOW" -ge 6 ] || [ "$H" -ge 22 ] ||')   # D98 (2026-11-14): Saturdays and Sundays all day
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
