p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    wsl bash tools/env/_py14.sh jprune_b.py commands --seed 0 [--point t1]                     # the scoring lines (jends_eval.py score --full --base --tag) of the read-outs snap / primary / attach",
              "    wsl bash tools/env/_py14.sh jprune_b.py commands --seed 0 [--point t1] [--groups]          # the scoring lines (jends_eval.py score --full --base --tag, `$R` = the problem folder) of the read-outs snap / primary / attach / unsnapped")
s = s.replace("A pass runs BOTH models on the same batch: a hook that does not fire once per forward or changes nothing stops the dump (E345's `Watch`).\n\"\"\"",
              "A pass runs BOTH models on the same batch: a hook that does not fire once per forward or changes nothing stops the dump (E345's `Watch`). On the CPU the pass takes about 0.7 s per frame (655 s per point of 873 frames, 2 threads); B's section stage needs the single-threaded\nsolve of `cpu_solve_fix` there (the threaded oneMKL LU of this box fails for 512 x 512). B' on the CPU matches the stored dump of the GPU as sets (the order of equal-valued nodes differs) and the scorer's numbers to 6e-5 in topo F1 (`check`, notes in the E352 card).\n\"\"\"")
open(p, "w", encoding="utf-8").write(s)
