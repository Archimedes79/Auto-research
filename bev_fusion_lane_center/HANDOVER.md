# HANDOVER - 2026-11-10 12:00

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms (B = baseline only). **D57 + D61: this (Spurmitte lane) chat has priority.** GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). **D60: when Mirko says he is away, work alone.** No TEST, push, mail, shutdown. Do not edit `autogt/` (GT chat).

## Results (cards E275 - E283). System report: https://claude.ai/artifact/9ShDVSELV2gi6xLLcGmpVF (copy in Dropbox AptivFolder, 2026-11-10)
- **E275 KEEP** hard close-node step; **E279** soft step 1.25 m KEEP (**open for Mirko, no D-id:** adopt it instead of the hard read-out, changes B, D58). E276 t TIE / s failed; E277 CUT; E280 / E281 / E282 closed by CPU gates; D64 long-link threshold stays 0.5.
- **E283 (analysis card, CPU only; `experiments/E283_junction-error-analysis/experiment.md`):** junction blur = unknown entry -> exit PAIRS (oracle true pairs 0.88, all pairs 0.53, network 0.71); geometry of clean ends predicts pairs (AUC 0.97 - 0.995, ~35 junction groups only). Largest graph loss = local 1 - 2.5 m offset in the head's own map (11.5 % of GT length). Five CPU levers closed by written gates.
- **D71 (Mirko 2026-11-10): junction entry (stop) AND exit points as late as possible but where the network can see them.** E283 section 5 measured it (`temp/e284/p17_end_offset.py`): the map's point is already about where the paint ends (paint support peaks there, 0.57 at entries), camera visibility is flat (0.65) so it cannot place a point, the network puts ends 0.2 - 0.6 m early and rarely late. Main defect is not the position but the flag: missing at 25 - 33 % of GT ends, about half of the flags sit at no GT end. Moved GT points are not the first lever. Direction: node heading median error 3.4 deg, ~99 % of matched junction ends within 30 deg; wrong-way rate on lanes without traffic not measured.
- Options, none decided: end keypoint head (GPU, needs a card), pair decision with geometry in the link head, CPU scorer oracle (GT ends + pair classifier + one curve per pair) FIRST. Open and unmeasured: map point vs painted stop line / crosswalk; wrong-way headings.

## State of the machine
Lane chain done; nothing of mine runs. Lane GPU job needs the other chats told first (GT chat: told of the p17 CPU pass). GT stores live on D: (links), l5 `db288acca5` merged, default store `0702a065c6` (l4). WSL page-cache ENOMEM: `temp/e276/reclaim.py`. Scratch: `temp/e284/` (tags `lcr/e284-scratch-1`, `-2`).

## If continuing
Ask Mirko: CPU scorer oracle with clean ends, or the wrong-way / stop-line measurements, or a card for the end keypoint head. Nothing pushed.
