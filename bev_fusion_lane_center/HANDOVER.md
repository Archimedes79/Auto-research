# HANDOVER - 2026-11-10 15:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only). **EXPERIMENTS PAUSED (D74, Mirko 2026-11-10): finish the system rebuild completely first, then experiments here. Start nothing (no card, run, GPU job, oracle) until he says the rebuild is complete.** D57 + D61 (lane first) only order the experiments after that. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: when Mirko says he is away, work alone. No TEST, push, mail, shutdown. Do not edit `GtGenerator/` (GT chat).

## Results (cards E275 - E283). System report: https://claude.ai/artifact/9ShDVSELV2gi6xLLcGmpVF (copy in Dropbox AptivFolder)
- **E275 KEEP** hard close-node step; **E279** soft step 1.25 m KEEP (**open for Mirko, no D-id:** adopt it instead of the hard read-out, changes B, D58). E276 TIE / failed; E277 CUT; E280 - E282 closed; D64 long-link threshold stays 0.5.
- **E283 (analysis card, CPU only, sections 1 - 7):** junction blur = unknown entry -> exit PAIRS; largest graph loss = local 1 - 2.5 m offset in the head's own map; five CPU levers closed by written gates.
- **D71 (junction entry AND exit points as late as possible but where the network can see them), section 5:** the map's point is already about where the paint ends; the network puts ends 0.2 - 0.6 m early; the defect is the flag (missing at 25 - 33 % of GT ends, ~half of the flags at no GT end), not the position.
- **Direction (section 6):** ~99 % of matched nodes point the right way (wrong way 0.7 %).
- **Scorer oracle (section 7, rule written before, NOT PASSED):** clean GT ends + true pairs: junction F1 +0.029 / +0.049 (needed +0.03 on both seeds) but topo F1 +0.10, lanes found +0.22, Q +0.07; with classifier pairs Q +0.04, topo F1 +0.04, junction F1 -0.01 / +0.01. By hand the other headlines move a lot, junction precision does not.

## Open for Mirko (wait with the experiments)
1. By which numbers is an end keypoint head judged: topo F1 / lanes found / Q (my recommendation; junction F1 as a guard) or junction F1? 2. Soft step 1.25 m instead of the hard read-out. 3. New lane version l6 of the GT chat (its findings: `GtGenerator/REVIEW_2026-11-10.md`; nothing starts before he says so). Then: CPU oracle with the measured end error, map point vs stop line, card for the head.

## State of the machine
Nothing of mine runs. GT stores on D: (links), default store `0702a065c6` (l4). WSL page-cache ENOMEM: `temp/e276/reclaim.py`. Scratch: `temp/e284/` (tags `lcr/e284-scratch-1`, `-2`, `-3`). Nothing pushed.
