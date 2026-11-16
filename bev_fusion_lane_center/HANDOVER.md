# HANDOVER - 2026-11-16 21:00 (auto operation, D86 - D112)

**SHUTDOWN (D111 / D112, Mirko): E362 is done (INCONCLUSIVE, committed bf1e7c6e); the report chat sends report 16g (E362 in it) by 21:40; then the lead runs `shutdown /s /t 60` (no run.py job, no fork turn, all committed; latest 23:00; he can move or cancel it). If this line is still here after a restart: nothing was running, start from run.py only.**

New chat: "Read HANDOVER.md and continue". Results: cards, `LEDGER.md`, `decisions/`.

**Rules:** D86: detection up, SIMPLE, LITTLE compute (ms, GMACs, parameters in every result), E2E. D87 TIE kept. D88 only E2E candidates. D94 system of record = E324 s. D95 ONE TEST of E324 s (E343, ON HOLD, his word). D100 two GPU slots, D101 `--workers 4`. D106: similar performance -> simpler AND faster wins. No mail / install / push unless asked.

**D109 (Mirko): NOT a pure E2E solution; accept only E2E.** E2E read-outs of E324 s (VAL topo F1 seed 0 / 1): rule wiring PRIMARY 0.6912 / 0.6948; **learned wiring head W (E358, KEEP+, +545 params, +2.0 ms host side) 0.6931 / 0.6964** (reported E2E result by the lead's default, revertible); **E2E stack W + Matrix-NMS layer, no close-node step (E363, TIE, dup guard broken, layer ms unmeasured) 0.6820 / 0.6854**; strict attach-only 0.6616 / 0.6646; snapped record 0.6981 / 0.7010 has a snap rule step = not accepted. Hard steps R1 - R8, T1 - T3: `experiments/E2E_gap_table.md` (Douglas-Peucker is NOT a delivered step).

**D110 = OPEN QUESTION** (his criticism 'solution space that does not exist'): proposal E2E system as reference, no cut for scoring below a rule-step system, adopt ties; answer awaited, adopt-on-tie NOT applied. **Token design: E357 / E359 / E361 FAIL, E362 (16000 steps) INCONCLUSIVE: g1 0.533 / 0.511, G2 0.711 / 0.713, type head the weak output, schedule saturated; T1 (predicted nodes) = his decision (D105 / D111).** Two parallel trainings on the card gave 12 CUDA 'unknown error' resumes and one GPU stall (nvlddmkm 13): use one slot or a supervisor.

**Pushed 2026-11-16:** main b1b4f579 + 163 `lcr/*` tags (held back `autogt/prep-v13-l6-2026-11-10`); later commits local. PROBLEM.md line proposed, needs his grant.

**E364: cover drop R4 carries 0.027 - 0.030 topo F1; the pair stage is limited by END recall (67 - 68 % of true connectors have both ends).** **Next (E2E, one card each, critic first):** layer duplicates + ms of layer / stack; soft cover drop (bar 0.027); end recall; pair run on B' nodes. Promote `temp/e346`, `temp/e358`, `temp/e364`, `temp/jr/*.py` into code folders with tests.

**Open for Mirko (report ch. 7):** D110; W vs rule wiring; stack vs W-only; E343 a / b; E348; E353; T1; E352 trained stage; E317 hold; `.wslconfig`.
