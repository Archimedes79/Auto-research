# E<id> <short title>

One file per experiment, in the experiment folder; it grows from card (P2) to detail plan (P3 start) to summary (P3 end). The machine-readable part (arms, seeds, metric, band, rule) lives next to it in the experiment file the tools read.

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

## 2. Detail plan (written when the loop reaches the card)
What would make it fail, in our own words: ...

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | | | spec test passes; gradient reaches the new parameters | |
| train | arms, seeds, configs | launcher command | done flag | |
| evaluate | registry names, split, comparison | evaluation entry point | metric card | |
| decide | the rule above | decision tool | verdict in the ledger | |

## 3. Summary (written after the decision)
- **Runs** (ledger ids, code tags) and **decision** (KEEP / CUT / TIE / INCONCLUSIVE, by the rule as written):
- **Setup** (architecture drawing, the one change, controls):
- **Numbers** (from metric cards; split and frames named; with controls; ms per frame):
- **Images**:
- **What it says** (about the hypothesis, why; negatives stay):
- **Weaknesses of this experiment** / **open points and next steps**:
