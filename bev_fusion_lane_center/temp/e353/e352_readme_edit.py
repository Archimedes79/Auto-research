p = "evaluation/README.md"
s = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(s) if l.startswith("| `jprune_table.py`"))
row = ("| `jprune_b.py` | E352 step 0 (no training): the E345 hook edits of B's pyramid trunk applied to B's OWN QGraph (`runs/e273_p_s<S>`: nodes / links / assignment) and to the junction system in one pass over the 873 VAL frames: `plan` (the 9 point sets: null, T1, T2, 3 random draws of each with the same "
       "channel counts, saved GMACs by `jprune_cost`, the point-set file), `dump` (`runs/e352_<point>_s<S>/`: link to the unmodified E324 s checkpoint, `ends_val.pkl` of the junction system on the edited pyramid, `predfull_<point>_s<S>.pkl` = B' in the format of `temp/e276/predfull_e273_p_s<S>.pkl` (`qpredict.predict` + "
       "`qgraph_readout.dump_full`), `point.json` last; `--groups` = whole GroupNorm groups), `check` (the empty set against the stored B dump and the stored E345 null), `commands` (the `jends_eval.py score --full --base --tag` lines for snap / primary / attach / unsnapped), `selftest`. No digested file is edited | "
       "new (imports `jprune_probe`, `jprune_cost`) | `jprune_b.py selftest`, `tests/test_jprune_b.py` |")
s.insert(i + 1, row)
open(p, "w", encoding="utf-8").write("\n".join(s))
p = "tools/maintenance/run_all_tests.sh"
t = open(p, encoding="utf-8").read()
k = 'test_jprune_probe.py\n'
t = t.replace(k, k + 'run "E352 step 0: B-side pruning probe (point table, shim, the empty set = the stored dumps, edited passes, point folder, scoring lines; CPU, about 10 minutes)" test_jprune_b.py\n', 1)
t = t.replace('jprune_probe.py selftest\n', 'jprune_probe.py selftest\nrun "E352 step 0 probe: point table on a random-init model, whole groups, the shim around B\'s QGraph (random init, CPU, no data)" jprune_b.py selftest\n', 1)
open(p, "w", encoding="utf-8", newline="\n").write(t)
