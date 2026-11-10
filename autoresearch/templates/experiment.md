# E<id> <short title>

One file per experiment, in the experiment folder; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). The machine-readable card is the block in section 0 of this file (validated against `autoresearch/schemas/experiment.schema.json` by `experiment.py check`, read by `decide.py`); there is no separate `experiment.json`.

## 0. The card (what the tools run)
Fields: id / card / title; kind (variant = configuration only, block = new code); deterministic (true when the arms share one checkpoint: no training seed, the paired scene-bootstrap interval decides); metric, direction, band (required; 0.010 unless the project measured another seed spread, then cite it); prediction {value, vs: arm letter}; arms {letter: recipe or command, base, set, seeds, run_ids, cards or names}; control; rule (clauses: metric, vs, min_effect or max_loss, band, subsets, at_least, note); budget_gpu_h, kill; evaluate {split, subset, compare}; sources (optional: `["<topic slug>/<note id>"]`, the research notes it tests; `decide.py --write` fills their result line). The rule is fixed before the run: `run.py` stores its hash in the run record and `decide.py` refuses a rule edited after the launch (`--by-hand "reason"` labels a verdict that goes around it).

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
 "arms": {"a": {"recipe": "...", "run_ids": ["E<id>.a_s0", "E<id>.a_s1"]}, "c": {"recipe": "...", "run_ids": ["E<id>.c_s0", "E<id>.c_s1"]}},
 "control": "c",
 "rule": [{"metric": "Q", "min_effect": 0.01}],
 "budget_gpu_h": 0.0,
 "kill": "loss@500 > 2.0",
 "evaluate": {"split": "val"}
}
```

`run_ids` (the planned ledger ids of the arm, list them BEFORE the first launch) is what registers the rule: `run.py` finds the card through it and stores the hash of the rule block in the run record; `decide.py` then checks that every listed run finished (ledger status ok, not repeating, started with a card), refuses a rule edited after the launch, and writes its token on the ledger rows of the variant arm. A card without `run_ids` still decides, but is labelled `rule not registered at launch`. `cards` or `names` (the metric cards of the arm, added after the evaluation) are what `decide.py` reads; an arm without them is not decided. `kill` is a trainsuite rule `key@step > value`.

## 1. Card text (written in P2, checked by ar-critic, mode cards; one page)
The prediction, the decision rule, the arms and controls, the budget and the kill criterion are the block above: write each once, there, and do not copy them into prose (a rule written twice drifts). Evidence and results belong in section 3.

- **Statement** (falsifiable, one sentence):
- **Mechanism** (why it should work):
- **Inputs** (ready / to build: data, code blocks, tests):
- **Starts from** (the version or run it changes):
- **Lens / group / sources / status** (one line: lens, problem group, note and K ids; status proposed / accepted / scheduled / confirmed / refuted / closed):
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
