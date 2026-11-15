import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e343", "run_rehearsal_smoke_r4.sh")
s = open(p, encoding="utf-8").read()
old = ' --expect-code-digest 0123456789abcdef0123456789abcdef\necho "exit $?"\necho "=== done"'
new = '\necho "exit $?"\necho "=== (b) a smoke card cannot pin the code (the rehearsal card must be a full VAL card)"\n$P all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --boot 50 $M --b-dump "$D/predfull_e273_p_s{seed}.pkl" --control-ends-dump "$D/ends_e312_p_s{seed}.pkl" --control-card "$D/cards_e312/card_s{seed}.json" --expect-code-card "$D/cards_e324s/card_s0.json"\necho "exit $?"\necho "=== done"'
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
