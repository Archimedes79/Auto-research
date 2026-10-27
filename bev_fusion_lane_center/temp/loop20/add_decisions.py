p = "experiment_execution/DECISIONS.md"
t = open(p, encoding="utf-8").read()
new = (
"| D18 | per-metric noise bands: `decide.py` and the cards use one band 0.010 (S / Q) for every metric. Four paired seeds (E240) measured the seed SD: Q 0.0025 (paired 0.0043), lanes found 0.004 (paired 0.0059), R@1 far 0.003 (paired 0.0062), topo F1 0.001, F1@1.0 0.001. Change the bands (tool + `project.json` / card fields, protected)? | (a) keep one band 0.010 / (b) per-metric band = 3 paired SDs, taken from a table in `project.json` (Q 0.013, lanes found 0.018, R@1 far 0.019, topo F1 0.005) / (c) per-metric band from the card, default 0.010 | (c): the card names the band and cites the E240 table; the tool keeps its default | `experiments/E240_measurement-floor-seed-noise/seed_sd.txt`, four pairs, VAL 873; the standard error of an SD with n = 4 is about 40 %, so bands are estimates |\n"
"| D19 | machine lease between chats: loop 20's timing window (E253) was agreed by message with the BEV-system chat and overrun by its 7-hour training; a timing or an exclusive run needs a lock that every launcher honours (`run.py` is the tools chat's) | (a) leave, agree by message / (b) a lease file `autoresearch/state/machine_lease.json` (owner, start, end, exclusive yes / no) that `run.py` and the other chats' launchers check / (c) only the timing driver takes a lock and refuses when a second GPU process exists (done in `temp/loop20/run_e253.sh`) | (b), with (c) as the minimum | E243 (invalid session), E253 (blocked), the loop-19 night (load 30 - 64, a 4800-iteration fine-tune at 1 s per iteration) |\n"
)
i = t.index("\n## Decided")
t = t[:i].rstrip("\n") + "\n" + new + t[i:]
open(p, "w", encoding="utf-8").write(t)
