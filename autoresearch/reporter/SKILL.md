---
name: reporter
description: Write, build, check and pack a paper-style, engineer-readable report with the standalone tool reporter/reporter.py. Use whenever a result report, status report or documentation package for a system is requested.
---
# Writing a report with the reporter
Goal: **an engineer who does not know the system understands what it is for, believes the numbers, is curious about the pictures, and can repeat the result.** The tool does the mechanics (`README.md`); the chapters and what each must hold are the prompts that `reporter.py new` writes.

## Quality bar (a good extended paper, not a log)
- A teaser and an abstract first: one real example as Figure 1; four sentences with the headline number next to its baseline.
- Captions stand alone: what is shown and what to look at. One idea per figure, the same colours for the same things, baseline and winner side by side on the same frame.
- Tables: baseline row first, winner in bold, split and denominator in the caption, units in the header, controls next to the result.
- Limitations are a section: failure modes with counts, what is in-sample, what was not measured; negative results stay. Examples are chosen by a written rule, not as favourites.

## Procedure
1. `new` once, then write `parts/` and `report.json`, in plain words first, then the details.
2. Read the primary sources (cards, result files, decisions), not summaries. A number is a `value` (file + key) or a literal of a card; list those files in `sources`, the kept scripts in `utilities`.
3. `build`, `check`; fix or explain every unfound number; `build --strict` and `check --strict` pass (a PENDING mark only in a marked draft).
4. A fresh reviewer reads it against the primary sources; apply, rebuild.
5. `pack`; send only on the owner's word.

## Rules that never bend
- No number from memory, no rounding that changes a verdict.
- VAL and TEST are never mixed; no TEST number without the owner's recorded permission; anything fitted or chosen on the evaluation set is labelled in-sample.
- A timing carries its protocol (device, load, batch, frames); numbers of different protocols are never subtracted or turned into percentages.
- Every report holds an architecture drawing of the current system (the owner, 2026-11-25); rule-based and hand-tuned parts are named as such.
