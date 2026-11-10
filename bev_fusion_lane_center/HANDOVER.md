# HANDOVER - 2026-11-10 14:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: when Mirko says he is away, work alone.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat).

## Results (cards E275 - E283). System report: https://claude.ai/artifact/9ShDVSELV2gi6xLLcGmpVF (copy in Dropbox AptivFolder)
- **E275 KEEP** hard close-node step; **E279** soft step 1.25 m KEEP (**open for Mirko, no D-id:** adopt it instead of the hard read-out, changes B, D58). E276 t TIE / s failed; E277 CUT; E280 / E281 / E282 closed; D64 long-link threshold stays 0.5.
- **E283 (analysis card, CPU only, sections 1 - 7):** junction blur = unknown entry -> exit PAIRS; largest graph loss = local 1 - 2.5 m offset in the head's own map; five CPU levers closed by written gates.
- **D71 (Mirko 2026-11-10): junction entry (stop) AND exit points as late as possible but where the network can see them.** Section 5: the map's point is already about where the paint ends; camera visibility is flat; the network puts ends 0.2 - 0.6 m early, rarely late; the defect is the flag (missing at 25 - 33 % of GT ends, ~half of the flags at no GT end), not the position. Moved GT points are not the first lever.
- **Direction (section 6):** ~99 % of matched nodes point the right way (wrong way 0.7 %, also on opposite lanes), errors at range >= 30 m and at split / merge / start / end points.
- **Scorer oracle (section 7, rule written before, NOT PASSED):** clean GT ends + true pairs: junction F1 +0.029 / +0.049 (needed +0.03 on both seeds), but topo F1 +0.10, lanes found +0.22 / +0.24, Q +0.07 / +0.08; with the classifier's pairs Q +0.04 / +0.05, topo F1 +0.04 / +0.05, junction F1 -0.01 / +0.01. Reading of the rule: no GPU hour for an end head on this evidence; by hand the other headlines move a lot (junction precision does not). Whether an end head is judged by junction F1 or by topo F1 / lanes found / Q is Mirko's call.
- Options, none decided: end keypoint head (GPU, needs a card with its own rule), pair decision with geometry in the link head, CPU sensitivity of the oracle to the measured end error, map point vs painted stop line / crosswalk (not measured).

## State of the machine
Nothing of mine runs. Lane GPU job needs the other chats told first. GT stores live on D: (links), default store `0702a065c6` (l4). WSL page-cache ENOMEM: `temp/e276/reclaim.py`. Scratch: `temp/e284/` (tags `lcr/e284-scratch-1`, `-2`, `-3`).

## If continuing
Ask Mirko which of the options; E283 sections 5 - 7 first. Nothing pushed.
