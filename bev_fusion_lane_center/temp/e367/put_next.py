p = "experiments/E367_e365-capacity-balance-schedule-arms/experiment.md"
s = open(p, encoding="utf-8").read()
a = "(a_wide and c_long follow)"
assert s.count(a) == 1
txt = ("**Not run yet (2026-11-17 09:15, the day ended with Mirko's wrap-up order): a_wide and c_long.** To start them in one step (from bev_fusion_lane_center, one GPU slot each, behind any running job; the card's texts as --hyp / --pred / --rule):\n"
       "```\n"
       "wsl python3 ../autoresearch/tools/run.py --id E367.a_s0 --seed 0 --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 130 --hyp \"E367 a_wide\" --pred \"see card\" --rule \"E367 card section 0\" -- bash tools/env/_pyr.sh jtoken_train.py --name e367_a_s0 --seed 0 --ident e292_a --d 96 --layers 4 --ffn 192 --steps 12000 --bs 8 --lr 1e-3 --workers 4\n"
       "wsl python3 ../autoresearch/tools/run.py --id E367.a_s1 --seed 1 --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 130 --hyp \"E367 a_wide\" --pred \"see card\" --rule \"E367 card section 0\" -- bash tools/env/_pyr.sh jtoken_train.py --name e367_a_s1 --seed 1 --ident e292_a --d 96 --layers 4 --ffn 192 --steps 12000 --bs 8 --lr 1e-3 --workers 4\n"
       "wsl python3 ../autoresearch/tools/run.py --id E367.c_s0 --seed 0 --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 260 --hyp \"E367 c_long\" --pred \"see card\" --rule \"E367 card section 0\" -- bash tools/env/_pyr.sh jtoken_train.py --name e367_c_s0 --seed 0 --ident e292_a --steps 24000 --bs 8 --lr 1e-3 --workers 4\n"
       "```\n"
       "Read-out per head (as the b arms): `jtoken_eval.py dump --run <R> --seed <S> --split train --every 3`, `dump --split val`, `sweep --split train --every 3` (choose by the E366 rule on that head's table, write the cell here BEFORE the VAL score), then `score --link-thr .. --keep-thr .. --no-none-gate`. A GPU crash supervisor is advisable (the pattern of temp/e367/sup.sh: relaunch the run.py id while no job of that id is alive and the log has no 'done' line). The time of the d 96 head on a quiet host (G-cost) comes first for a_wide.")
s = s.replace(a, txt)
open(p, "w", encoding="utf-8").write(s)
print("ok")
