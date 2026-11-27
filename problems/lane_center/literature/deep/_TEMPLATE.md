# <ID> - <short name>: full-text extraction

Copy this file to `deep/<ID>_<slug>.md` (<ID> = home note id from `../CATALOGUE.md`, slug = lower-case ASCII, hyphens). Fill every section; write "not found" instead of leaving one empty. English, paraphrase, keep numbers and equations as facts. Delete these instructions.

| field | value |
|---|---|
| paper | title, authors, venue year, arXiv id, links actually opened |
| axis, tier, K ids | from `../papers.csv` / `../CATALOGUE.md` (A1..A9; core = 1) |
| level-1 note(s) | links to `../papers/<ID>_*.md` (all notes of this paper) |
| code extraction | link to `../code/<repo>.md` or "no repository" |
| sources read | list: page, version (v1/v2), what part (abstract / HTML sections / appendix / supplement), date |
| reading marks | per section: **checked** (page seen), **digest** (seen through a summarising tool), **memory** (not seen) |
| extracted by, date | agent / session, YYYY-MM-DD |

## 1. Problem and claim
Three to five lines: what is predicted from what, which gap the paper closes, the one claim that matters.

## 2. Method as built
- **Pipeline / architecture:** blocks in order with tensor shapes and channel counts (draw a small ASCII chain).
- **Inputs, outputs:** exact formats (coordinates, frames, number of elements, padding rules).
- **Equations:** the ones needed to re-implement, numbered as in the paper (E1, E2 ...), symbols explained.
- **Losses:** each term, weight, normalisation; assignment rule (Hungarian? nearest? threshold?).
- **Labels / ground truth:** how targets are built from the data set (often missing in the paper text; take it from code).
- **Training recipe:** optimiser, lr and schedule, epochs, batch, augmentation, hardware and time, warm-up / curriculum.
- **Inference:** thresholds, NMS or merge rules, post-processing (these are the hidden rule steps), caps, test-time tricks.

## 3. Numbers
- Main result table (benchmark, protocol, metric, value, baselines) - copy only what you saw, with the table number.
- Ablations that decide transfer (what was removed, what changed).
- Compute: parameters, ms or FPS, GPU, batch size, input size.
- Caveats: protocol differences, split, range, tolerance (numbers of different papers are not comparable).

## 4. Paper versus code
Where released code differs from the text (defaults, missing losses, extra filters). "No code" if so.

## 5. Transfer to our pipeline
| question | answer |
|---|---|
| what it needs from us | frozen cached BEV 200 x 200 at 0.5 m (39 ch)? ego poses? SD map? image features? lidar? |
| what it gives us | which stage of the chain (A1..A9), which output |
| conflicts | Hungarian matching, argmax / tokens on the training path, perspective image, trainable encoder, lidar, global pose, mm-stack, GPL |
| reuse mode | as-is / adapt code / re-implement from description / idea only |
| effort | S (hours) / M (days) / L (weeks), with the blocker |
| expected gain and risk | tied to our open items (see `../OVERVIEW.md` section 3) |
| feeds which experiment / K id | proposed K id if it is a new technique (register first, then use) |

## 6. Not found / open
What the paper does not say and we would need (hyper-parameters, supplement, failure cases).

## 7. Our experiments that touch it
E ids and one-line verdicts, so nobody repeats them.
