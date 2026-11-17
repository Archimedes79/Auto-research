p = "experiments/E366_e365-decode-selected-on-train/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep('"budget_gpu_h": 0.0,', '"budget_gpu_h": 0.2,')
rep('"kill": "a read-out is not finite; no GPU job (CPU, niced, 2 threads)",',
    '"kill": "a read-out is not finite; GPU only for the four short dumps (run ids E366.dump_s<S>: the S1 heads re-dumped in the new format, TRAIN every 3 and VAL, a few minutes), the sweep and the scoring on the CPU (niced, 2 threads)",')
rep("`jtoken_eval.py sweep` on the TRAIN sample (cards under runs/e365_s1_s<S>/token_sweep_train.json)",
    "`jtoken_eval.py dump --split train --every 3` and `dump --split val` (new format: per-candidate handle scales, float32 none probabilities; the E365 dumps were kept as token_*_v1.pkl), `jtoken_eval.py sweep --split train --every 3` (runs/e365_s1_s<S>/token_sweep_train_every3.json)")
rep('"run_ids": ["E366.sweep_s0", "E366.sweep_s1", "E366.val_s0", "E366.val_s1"]',
    '"run_ids": ["E366.dump_s0", "E366.dump_s1", "E366.sweep_s0", "E366.sweep_s1", "E366.val_s0", "E366.val_s1"]')
rep("The paired scene-bootstrap interval dec - ref is reported per seed.",
    "The paired scene-bootstrap interval dec - ref is reported per seed (`paired_vs_record`: the record decode scored on the SAME re-dump; its difference to the stored E365 record numbers from the v1 dumps is reported, it is the float16 / float32 effect of the dump format).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
