p = "experiments/E375_champion-per-class-decode/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep("Card of 2026-11-17 (version 1, 08:15 system clock, before any sweep;",
    "Card of 2026-11-17 (version 2, 08:35 system clock, after a short fresh critic (cards: REVISE, 6 findings, taken in: lanes found guarded and the gate re-checked at the new cell; the mechanism rewritten to what the scorer's junction F1 measures; the tools must carry junction counts and a paired interval; ADOPT needs the interval; the per-class vs re-tuned single threshold wording; ties and references), before any sweep;")
rep('"prediction": {"value": 0.03, "vs": "ref"},', '"prediction": {"value": 0.02, "vs": "ref"},')
rep("among the cells whose mean TRAIN dup_frac <= 0.05, take",
    "among the cells whose mean TRAIN dup_frac <= 0.05 and whose mean TRAIN lanes found is at most 0.01 below the champion's cell (0.8110 on the two E367 tables), take")
rep("ties to the cell nearest the champion's cell (short threshold, then long, then keep).",
    "ties to the cell nearest the champion's cell (short threshold, then long, then keep; then the higher threshold).")
rep("at the chosen cell against the champion's cell on the same head: topo F1 not lower by more than 0.005, F1@1.0 not lower by more than 0.005, AND junction F1 higher by at least 0.03;",
    "at the chosen cell against the champion's cell on the same head: topo F1 not lower by more than 0.005, F1@1.0 not lower by more than 0.005 (both deliberately stricter than the band 0.010), lanes found not lower by more than 0.01, AND junction F1 higher by at least 0.03 AND the lower end of the paired scene-bootstrap 95 % interval of the junction F1 difference above 0; AND at the chosen cell the D113 gate floors (E365 section 0) and the E373 replacement tie conditions (topo F1 >= 0.6518 / 0.6650, F1@1.0 >= 0.8281 / 0.8310, lanes found >= 0.6698 / 0.6776) hold on both fresh heads;")
rep("If the chosen cell IS the champion's cell, the verdict is NO by construction (reported: 'the per-class grid prefers the current cell').",
    "If the chosen cell IS the champion's cell, the verdict is NO by construction (reported: 'the per-class grid prefers the current cell'). If the chosen cell has short = long, it is reported as 'a single link threshold re-chosen by the junction tie-break' (three tuned output thresholds); otherwise G-simple holds with FOUR tuned output thresholds (short link, long link, keep, none gate off), all on the output and named; on ADOPT the named-hard-steps line of the champion (D114) is updated. The fresh heads' TRAIN dumps are not swept before the verdict. `E375.val_s0` also scores the champion's cell 0.2 / 0.2 / 0.6 / off on the E367 seed-0 head (it was read at 0.2 / 0.7 only).")
rep("**Reported:** the full 36-cell TRAIN tables (topo F1, junction F1, dup_frac, F1@1.0, lanes found per cell);",
    "**Reported:** the full 36-cell TRAIN tables (topo F1, junction F1 / P / R and the predicted junction points per frame, dup_frac, F1@1.0, lanes found per cell); on the TRAIN tables the one-threshold steps from the champion's cell to the chosen cell (only long, only short, only keep changed); junction P / R and junction points per frame for every read-out;")
rep("**Prediction (guesses, with windows):** the chosen cell has a long threshold BELOW the short one (70 %); junction F1 on the fresh heads +0.02 (window -0.01 to +0.06); ADOPT-DECODE about 30 % (the 0.03 bar is high: the junction F1 of the record decode, the most conservative cell, was 0.24 - 0.25, so the room above the champion's 0.20 inside this grid is about +0.05 at most, and topo F1 must hold at the same time).",
    "**Prediction (guesses, with windows):** the chosen cell has a short or a long threshold ABOVE the champion's 0.2 (fewer false fork points) (65 %); junction F1 on the fresh heads +0.02 (window -0.01 to +0.05); ADOPT-DECODE about 20 % (the 0.03 bar is high and the topo F1, F1@1.0 and lanes-found guards push the other way: stricter cells lose recall).")
rep("- **Mechanism:** the champion's single link threshold treats short lane links and long connector links alike; long links are rare (about 2.5 % of the source-class rows) and their probabilities are lower (E365 diagnosis: 51 % delivered at the record decode); a class-specific threshold can trade lane-link recall and connector-link precision separately.",
    "- **Mechanism (what the scorer measures):** the scorer's junction F1 is class-agnostic: it counts the points of degree 3 or more of the densified, undirected delivered graph, matched one-to-one within 3 m, pooled over frames (`evaluation/lane_task/measures.py`); at the champion's cell it is limited by PRECISION (VAL fresh heads junction P 0.119 / 0.115, R 0.689 / 0.680; at the record decode P 0.175, R 0.421); a gain needs fewer false points of degree >= 3: a long threshold above the short one (fewer fan-out forks of connectors) or a short threshold above 0.2 (fewer duplicate lane forks); a single connector from a lane end gives degree 2 and no junction.")
rep("| option | `--link-thr-long` in `jtoken_eval.py` score / sweep / graphs (default = `--link-thr`: the current behaviour bit for bit) and a per-class grid for `sweep` (`--grid perclass`), tests | coder | tests green, committed |",
    "| option | `--link-thr-long` in `jtoken_eval.py` score / sweep / graphs (default = `--link-thr`: the current behaviour bit for bit), a per-class grid for `sweep` (`--grid perclass`) whose rows carry junction_F1 / junction_P / junction_R and the junction points per frame, `score --ref-cell` writing the paired scene-bootstrap interval of topo F1 and junction F1 against the champion's cell on the same dump, tests | coder | tests green, committed |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
