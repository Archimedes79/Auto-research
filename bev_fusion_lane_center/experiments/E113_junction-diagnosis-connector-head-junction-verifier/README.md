# E113_junction-diagnosis-connector-head-junction-verifier

**Junction diagnosis; connector head; junction verifier**

| | |
|---|---|
| experiment id | E113 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | P - with a classical step: learned, but needs a rule or geometric stage next to it |
| sparsification stage | SP3 graph assembly and junctions (links, connectors, assembler); SP4 keep and verify (precision gates, verifiers, probabilities) |
| building blocks | BB13 relation bias, graph transformer, message passing |
| loop | loop13 - log: [../../experiment_execution/history/indexes/LOOP13_INDEX.md](../../experiment_execution/history/indexes/LOOP13_INDEX.md) |

## Result

1.8 times too many junctions; verifier split / merge F1 0.385 / 0.428 to 0.437 / 0.504 (rule graph 0.714 / 0.637); connector head lowers it

## Techniques used (register K ids)

- K86: lane topology as a chain of graph expansions (autoregressive, quantised)  (checked loops 13, 15)
- K107: learning non-maximum suppression  (checked loop 13 (abstract))
- K09: learned duplicate removal as a relation module  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 24 small files from `e113_a`
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
