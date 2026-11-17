---
name: reporter
description: Write, build, check and pack a paper-style, engineer-readable report (abstract and teaser, introduction, system and architecture, required ground truth with images, experiments very short, quantitative and qualitative results with good cases and issues, winning systems, conclusion and next steps, how to repeat, utilities) with the standalone tool reporter/reporter.py. Use whenever a result report, status report or documentation package for a system is requested.
---

# Writing a report with the reporter

The goal: **an engineer who does not know the system understands what it is for, believes the numbers, is curious about the pictures, and can repeat the result.** The tool (`reporter.py`, see its README) does the mechanical part; this skill is the procedure and the quality bar. The model is a good extended paper (CVPR / ECCV style structure), not a log.

## What the best reports and papers do (and we copy)
- **A teaser and an abstract come first.** One real, striking example (input and output side by side, or before / after) as Figure 1, and an abstract of four sentences that already contains the headline number next to its baseline (as in the original attention, ResNet, DETR papers: the claim is on page one). Distill.pub and Tufte: the picture carries the argument, the text explains how to read it.
- **Captions stand alone.** Every figure and table has a numbered caption that says what it shows *and what to look at* ("the far lane on the right is missed"); a reader who only looks at the figures can follow the story.
- **One idea per figure; same things look the same.** One colour per kind of block in the architecture drawing, the same colours for ground truth / delivered / missed in every picture; qualitative results as small multiples: the same frame with baseline and winner side by side.
- **Tables are read, not decoded.** The baseline row first, the winner in bold, split and denominator in the caption, best and second best marked, units in the header. Controls next to the result (null, matched reference, an oracle as a sanity check, not as a bound).
- **Honest limitations are a section, not a footnote.** Failure modes with counts ("how often does this happen"), what is in-sample, what was not measured. Negative results stay in the experiment table (a short row each).
- **Reproducibility is a chapter** (model-card / ML-reproducibility-checklist style): data and versions, commands in order, seeds, the numbers to hit at each stage, checks that tell a rebuild is right, known pitfalls. Test it by following it.
- **Typography:** one column of readable width, generous whitespace, serif text with sans headings and tables, no decoration that carries no information.

## Procedure
1. **Scaffold or open** the report folder: `python reporter.py new <folder> --title "..."` (once), then work in `parts/`, `media/`, `report.json`.
2. **Read the sources, not summaries:** the result sections of the experiment cards, ledger rows, decisions. Every number goes into `values` (file + key path) or is typed only when it is a literal from a card; put the cards and result files into `sources` so `check` can find it. List the helper scripts that must be kept under `utilities`.
3. **Write in the order of the paper**, in plain words, then the details:
   1. *Abstract and teaser* (in `report.json`), then *Introduction: purpose and status*: the problem, what was built, the status in one line, the headline result with its baseline, a map of the report.
   2. *The system*: one drawing (svg, no red blocks), the chain, what is learned and what is a named rule or budget, the loss, parameters and time per frame; terms defined once.
   3. *Required ground truth*, **with pictures** of the raw data next to the derived labels; why each element is needed, how it is made, how to know it is right, its limits.
   4. *Experiments done, very short*: one row per step (idea, result, verdict).
   5. *Quantitative results*: numbered tables, controls, seeds, noise band; plots and videos.
   6. *Qualitative results: good cases and issues*: real examples chosen by a written rule (never hand-picked favourites), good ones first, then the failure modes with counts; baseline and winner side by side.
   7. *The winning systems*: each with architecture, headline numbers, parameters, time per frame, where the weights and the run folder are.
   8. *Conclusion and next steps*: three lines of what the work shows, what is still hard, the next steps very brief with their cost, decisions that wait for the owner.
   9. *A. How to repeat the result*; *B. Utilities* and *C. Sources* are generated.
4. **Build and gate:** `build` (figures, tables, PENDING count), `check` (every number found in a source); fix every unfound number or explain it in the text; then `build --strict` and `check --strict` must pass. A remaining PENDING mark is allowed only in a deliberately marked draft or skeleton.
5. **Review:** a fresh reviewer (not the author) reads the report against the primary sources before it goes out; apply the findings, rebuild.
6. **Pack** (`pack`), check the size limit, **send only on the owner's word**.

## Rules that never bend
- No number from memory; no rounding that changes a verdict; no "about" where the source has a value.
- Never mix validation and test numbers in one comparison; no test number without the owner's recorded permission; anything fitted or chosen on the evaluation set is labelled in-sample.
- Timing numbers carry their protocol (device, load, batch, frames); numbers of different protocols are never subtracted or turned into percentages.
- Rule-based or hand-tuned parts of a system are named as such in the architecture chapter.
- Videos are not embedded: they live in `media/` and in the zip, with a still image in the text.
- Keep the file small: images are recompressed; respect the package limit.

## Minimal commands
```
python reporter.py new r --title "X"
python reporter.py build r ; python reporter.py check r
python reporter.py build r --strict ; python reporter.py check r --strict
python reporter.py pack r
```
