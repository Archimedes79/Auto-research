import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_ensemble-distilled-head", "experiment.md")
s = open(p, encoding="utf-8").read()

def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)

# arm e -> e2 with the cost measure, new control arm l
rep('''  "e": {"recipe": "the ENSEMBLE ceiling, no training (reported, never a candidate for the champion: two heads): the mean of the two teacher heads' dumps per seed at the D115 cell (`ens_step0.py`, promoted into `jtoken_eval.py` or a new `token_readout.py` command before the run); MEASURED today: topo F1 0.6792 / 0.6827 (E - ref +0.0113 [+0.0074, +0.0145] / +0.0065 [+0.0022, +0.0108])", "run_ids": ["E379.ens_s0", "E379.ens_s1"]},''',
'''  "e2": {"recipe": "the plain ENSEMBLE at inference, no training: the mean of the two teacher heads' p_none and link probabilities on the same B' tokens, read at the D115 cell (`ens_step0.py`, promoted into `jtoken_eval.py` or `token_readout.py` as an `ens` command before the run); MEASURED 2026-11-17: topo F1 0.6792 / 0.6827 (E - ref +0.0113 [+0.0074, +0.0145] / +0.0065 [+0.0022, +0.0108]). It is the UPPER BOUND the distilled head chases and, if cheap enough, itself a champion candidate (D106 chooses the simpler by cost): its cost is MEASURED in the job E379.time_e2 (GPU, night window, quiet host, batch 1, bf16, the 24-frame protocol of `jtoken_eval.py timing`): the head alone, two heads one after the other, the two heads stacked in one batched forward (`torch.func.stack_module_state` + vmap: the head is launch-bound, a stacked forward should cost far less than 2 x), the whole path with each, the parameters (2 x 127,131 = 254,262 for the heads; the whole path 2.323 M -> 2.450 M against E324 s 2.741 M); stored numbers (NOT QUIET host, `runs/e365_s1_s0/token_timing_20261117T034328Z.json`): head 6.6 ms, B' 11.4 ms, whole path 16.6 ms, so a second head sequentially adds about 5 ms (whole path about 21.5 ms against E324 s 19.7 ms): the stacked variant decides", "run_ids": ["E379.ens_s0", "E379.ens_s1", "E379.time_e2"]},
  "l": {"recipe": "the matched-COMPUTE single head (the schedule control, E367's c_long at the champion's none weight; ONE seed, seed 0, like c_long): `jtoken_train.py --name e379_l_s0 --seed 0 --head-seed 2 --ident e292_a --steps 24000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25` = 2 x the champion's steps (the distilled student costs about 1.6 x the champion's step time: two teacher forwards on top of one student step): if d beats this, the gain is not 'more compute on the same head'; E367's a_wide (d 96, 4 layers: capacity, 1.3 - 1.6 x the head's cost) stays the optional capacity control and is NOT run here", "run_ids": ["E379.l_s0"]},''')
rep('"budget_gpu_h": 1.5,', '"budget_gpu_h": 2.4,')
rep('''  {"metric": "dup_frac", "vs": "ref", "max_loss": 0.01}
 ],
 "budget_gpu_h"''', '''  {"metric": "dup_frac", "vs": "ref", "max_loss": 0.01},
  {"metric": "topo_F1", "vs": "l", "max_loss": 0.002}
 ],
 "budget_gpu_h"''')
# verdict text additions
rep("the kill did not fire. **NO** = d - ref <= -0.005 on both seeds", "the kill did not fire; **and, on seed 0, d is not below the matched-compute control l by more than 0.002 (if l - ref >= +0.005 on seed 0 the schedule explains the gain: the verdict for d is then TIE unless d - l >= +0.003)**. **NO** = d - ref <= -0.005 on both seeds")
rep("**INCONCLUSIVE** (the seeds disagree", "**The ensemble e2 is decided SEPARATELY and by cost (D106: the simpler champion), never by this verdict:** it is reported next to d with its measured ms and parameters; if e2 - ref >= +0.005 on both seeds AND the measured whole path of the two-head ensemble (stacked variant) is at or below E324 s's 19.7 ms, the lead decides between 'one head + a teacher at training' (d) and 'two heads at inference' (e2) by the card's scorecard axes (simplicity, compute, performance, code); else e2 stays a ceiling. **INCONCLUSIVE** (the seeds disagree")
rep("**Cost:** training: two teacher forwards", "**Cost (the arms):** training d: two teacher forwards")
rep("1.1 GPU h, dumps 5 min each", "1.1 GPU h, the control l one 24000-step run (0.7 h on an idle GPU, 1.4 h shared), the timing job 5 min, dumps 5 min each")
rep("(the head 0.127 M parameters, about 3.7 ms of the 16.7 ms path at batch 1;", "(the head 0.127 M parameters, 6.6 ms median alone / about 5 ms of the whole path (16.6 ms) on the stored NOT QUIET timing;")
rep("The ensemble itself would cost +1 head (+3.7 ms, 20.4 ms against E324 s 19.7 ms): the reason to distil.", "The ensemble itself would cost +1 head (about +5 ms, whole path about 21.5 ms against E324 s 19.7 ms; stacked: to be measured): the reason to distil, and the reason the card measures it.")
open(p, "w", encoding="utf-8").write(s)
print("patched")
