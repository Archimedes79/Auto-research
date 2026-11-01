# E<id> <short title>

One file per experiment, in the experiment folder; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). The machine-readable card is the block in section 0 of this file (validated against `autoresearch/schemas/experiment.schema.json` by `experiment.py check`, read by `decide.py`); there is no separate `experiment.json`.

## 0. The card (what the tools run)
Fields: id / card / title; kind (variant = configuration only, block = new code); deterministic (true when the arms share one checkpoint: no training seed, the paired scene-bootstrap interval decides); metric, direction, band (default 0.010; a per-metric band cites the project's measured seed spread); prediction {value, vs: arm letter}; arms {letter: recipe or command, base, settings, seeds}; control; rule (clauses: metric, vs, min_effect or max_loss, band, subsets, at_least, note); budget_gpu_h, kill; evaluate {split, subset, compare}.

```json card
{
 "id": "E<id>",
 "card": "H<..>",
 "title": "...",
 "kind": "variant",
 "metric": "Q",
 "direction": "higher",
 "band": 0.010,
 "prediction": {"value": 0.0, "vs": "c"},
 "arms": {"a": {"recipe": "..."}, "c": {"recipe": "..."}},
 "control": "c",
 "rule": [{"metric": "Q", "min_effect": 0.01}],
 "budget_gpu_h": 0.0,
 "kill": "...",
 "evaluate": {"split": "val"}
}
```

## 1. Card (written in P2, checked by ar-critic, mode cards)
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
- **Starts from** (the version or run it changes):
- **Referee** (one line: the critic's verdict and a link to its report):

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
