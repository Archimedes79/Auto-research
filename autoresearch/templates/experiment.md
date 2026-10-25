# E<id> <short title>

One file per experiment, in the experiment folder; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). The machine-readable part lives next to it in `experiment.json` (validated against `autoresearch/schemas/experiment.schema.json`); the table below shows the same fields so a reader sees what the tools will run.

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E<id> / H<..> / ... |
| kind | variant (configuration only) / block (new code) |
| metric, direction, band | headline metric (e.g. Q), higher / lower, noise band of that metric |
| prediction | value, against arm <letter> |
| arms | <letter>: recipe or command, base, settings, seeds |
| control | <arm letter> |
| rule | clauses: metric, vs, min_effect or max_loss, band, subset, quantifier, split |
| budget_gpu_h, kill | ..., ... |
| evaluate | split, subset, compare |

## 1. Card (written in P2, checked by ar-referee)
| | |
|---|---|
| lens | state of the art / foundations / training and losses / past critic / system review / free |
| group | <problem group id> |
| sources | <note ids, K ids, experiment ids> |
| status | proposed / accepted / revise / rejected / scheduled / confirmed / refuted / open / closed |

- **Statement** (falsifiable, one sentence):
- **Mechanism** (why it should work):
- **Prediction** (metric, split, number, against which reference):
- **Decision rule** (result -> action; noise band; what counts as a tie):
- **Test design** (arms; controls: null / matched reference / equal budget / seed twin; budget; fidelity rung):
- **Cost** (GPU hours, coding days, risk) and **kill criterion**:
- **Inputs** (ready / to build: data, code blocks, tests):
- **Referee** (verdict and reason):

## 2. Detail plan (block experiments only; written by ar-planner when the loop reaches the card)
What would make it fail, in our own words: ...

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | | | spec test passes; gradient reaches the new parameters | |
| train | arms, seeds, configs | launcher command | done flag | |
| evaluate | registry names, split, comparison | evaluation entry point | metric card | |
| decide | the rule above | decision tool | verdict in the ledger | |

## 3. Summary (written after the decision; one line is enough for a variant)
- **Runs** (ledger ids, code tags) and **decision** (KEEP / CUT / TIE / INCONCLUSIVE, by the rule as written):
- **Setup** (architecture drawing, the one change, controls):
- **Numbers** (from metric cards; split and frames named; with controls; ms per frame):
- **Images**:
- **What it says** (about the hypothesis, why; negatives stay):
- **Weaknesses of this experiment** / **open points and next steps**:
