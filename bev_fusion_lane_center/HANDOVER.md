# HANDOVER - 2026-11-17 close-out (auto operation, D86 - D115)

New chat: "Read HANDOVER.md and continue". Results: cards, `LEDGER.md`, `decisions/`. **The computer was shut down at the end of 2026-11-17 on Mirko's order (D115); main pushed before.**

**CHAMPION (D113 mission; D114 + D115, lead, by the written rules; Mirko can overrule): the E365 token head trained with NONE WEIGHT 0.25 (E373 b_bal25), read with the PER-CLASS decode of E375 (link short 0.3 / long 0.2, node p_none < 0.6, none gate off).** 0.127 M parameters, 3 relative-geometry self-attention layers = the wiring, per-source successor softmax + none slot, 7-way type softmax, on the predicted nodes of the frozen lane head B' (top-K 512), 12000 steps (41 min). VAL fresh heads: topo F1 0.6679 / 0.6762 (plain B 0.626 / 0.625), F1@1.0 0.846, lanes found 0.72, junction F1 0.25 (B 0.27; 0.20 before E375), 24 vertices / 100 m (B 19); whole path 16.6 - 16.8 ms vs E324 s 19.7, 2.323 M vs 2.741 M parameters. Status 'staged, B' frozen'; named hard steps: top-K 512, allowed-pair gate, four output thresholds. No TEST (E343 on hold, needs his recorded OK). Scorecard: `experiments/E2E_champion_scorecard.md` (3.5 of 5).

**Done today:** E370 MISS (rule-distilled start does not survive the GT loss), E375 ADOPT-DECODE, S2 launch gate passed, S3 map hook (E374) built, report + winner folders (`winners/`, `reporter/`) sent and pushed.

**NEXT (written, not started):** (1) S2: five E368 runs READY (z_s0, r_s0 first, then z_s1, r_s1, z2_s0; `--timeout 90`, about 3 h GPU; the start line is in the card); (2) S4 (E372: the loss reaches B's node head; needs the lane chat's live-token path); (3) S3 (E374); (4) E376 long-class none weight; E367 a_wide / c_long (start lines in the card); (5) promote scratch scripts (temp/e346, e358, e364, temp/jr) into code folders with tests. GPU windows: nights 22-07, Mon-Fri 08:30-17:30; one GPU job family at a time with a supervisor (two parallel trainings cause CUDA 'unknown error').

**Open for Mirko:** W vs rule wiring; E343 a / b (TEST); E348; E353; E352; E317 / E318 / E319 holds; `.wslconfig`; PROBLEM.md line; E369's rule prior (diagnostic only); D110 rule 3.
