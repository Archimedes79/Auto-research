# HANDOVER - 2026-11-16 14:50 (auto operation, D86 - D111)

**SHUTDOWN POSTPONED (D111 / D112, Mirko): the shutdown command of 17:21 was aborted (17:22) because Mirko said 'Do one more Experiment'. ONE experiment runs: E362 (Fable; node-type head, 16000 steps, two seeds). When its result section + ledger notes are committed and no run.py job / fork turn is running: `shutdown /s /t 60` (target 21:30, latest 23:00; he can move or cancel it). Do not forget after a compaction.**

New chat: "Read HANDOVER.md and continue". Results: cards, `LEDGER.md`, `decisions/`.

**Rules:** D86: detection up, SIMPLE, LITTLE compute (ms, GMACs, parameters in every result), E2E. D87 TIE kept (E2E replacement kept and refined). D88 only E2E candidates. D94 system of record = E324 s. D95 ONE TEST of E324 s (E343, ON HOLD, his word). D100 two GPU slots, D101 `--workers 4`. D106: similar performance -> simpler AND faster wins. No mail / install / push unless asked.

**D109 (Mirko): NOT a pure E2E solution; accept only E2E.** E2E read-outs of E324 s (VAL topo F1 seed 0 / 1): rule wiring PRIMARY 0.6912 / 0.6948; **learned wiring head W (E358, KEEP+, +545 params, +2.0 ms host side) 0.6931 / 0.6964** (reported E2E result by the lead's default, revertible); **E2E stack W + Matrix-NMS layer instead of the close-node step (E363, TIE kept and refined, dup guard broken, layer ms unmeasured) 0.6820 / 0.6854**; strict attach-only 0.6616 / 0.6646; snapped record 0.6981 / 0.7010 has a snap rule step = not accepted. Hard steps R1 - R8, T1 - T3: `experiments/E2E_gap_table.md` (Douglas-Peucker is NOT a delivered step).

**D110 = OPEN QUESTION** (his criticism 'solution space that does not exist'); proposal: E2E system as reference, no cut for scoring below a rule-step system, adopt ties; his answer awaited; adopt-on-tie NOT applied. Token design (D111): E357 / E359 / E361 FAIL (E361 g1 0.491 / 0.484 by a hair), E362 text draft only.

**Pushed 2026-11-16:** main b1b4f579 + 163 `lcr/*` tags (held back `autogt/prep-v13-l6-2026-11-10`). Report versions sent: 16b; 16f (or later) to be sent ~16:00 by the report chat. PROBLEM.md line proposed, needs his grant.

**E364 (cover drop R4 carries 0.027 - 0.030 topo F1, dup_frac +0.07 without it; pair mask R6 costs 2.6 % of true connectors; the pair stage is limited by END recall: only 67 - 68 % of true connectors have both ends matched).** **Next work block (E2E, one card each, critic first):** (1) refine the layer's duplicates (E363 guard) + ms of layer / stack on an idle GPU; (2) soft replacement of the cover drop (bar 0.027); (3) end recall of the pair stage; (4) pair run re-trained on B' nodes; (5) R3 / R7 need new dumps. Promote `temp/e346`, `temp/e358`, `temp/e364`, `temp/jr/build_ergebnis.py` + `pack_paket.py` (used several times) into code folders with tests; the first GPU use of `jstack.py` starts with its fixture dump. Memory: `lane-e2e-day-d109-d111-2026-11-16.md`.

**Open for Mirko (report ch. 7):** D110; W learned vs rule wiring; stack vs W-only; E343 a / b; E348 exception to D106; E353 MACs-only; E362 / T1; E352 trained stage; E317 hold (g read-out 0.6905 / 0.6971); `.wslconfig`.
