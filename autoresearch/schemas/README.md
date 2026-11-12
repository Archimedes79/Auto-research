# schemas

JSON schemas of the two contracts between agents and code.

| file | describes | checked by |
|---|---|---|
| `experiment.schema.json` | the `json card` block of an `experiment.md`: id, metric, direction, band, prediction, control, arms, rule, evaluate | `tools/experiment.py check` |
| `card.schema.json` | a metric card the evaluator writes: split, frames, metrics, their direction, provenance | `tools/card.py check` |
