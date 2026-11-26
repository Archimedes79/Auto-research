# toy - the reference problem of the research system

A disc drifts through a noisy 32 x 32 picture; a small network marks its pixels. Small enough for the CPU in minutes, it goes through every
activity of the system: data and ground truth, training, evaluation (numbers, pictures, compute time), a verdict, adoption and a report. It is
the chain test of `autoresearch/research` (`python -m research.selftest`) and the template of a new problem.

## What a problem writes (and nothing else)
| file | what |
|---|---|
| `problem.json` | the steps (prepare, train, evaluate), the score `seg.iou` with its band, a guard, the control floor, the holdout split, the protected files |
| `dataset/` | the recipe of the raw data (`make_data.py`) and its manifest (shards with split, size, sha256) |
| `gt/` | the ground-truth generator: `raster.py` is the hashed part; the store is named by the signature (rasteriser, options, data manifest) |
| `net/` | `prep/` the shard reader (checks every shard's hash, guards the holdout), `model/` the network, `loss/` the loss |
| `training/` | `train.py` on `trainkit`, `configs/base.json` (every setting an arm may change), the drawing of the system |
| `evaluation/` | `adapter.py`: sets, source, subject, metric with additive counts, the overview picture, the timing |
| `tests/` | `test_toy.py`: known answers (asymmetric), the refusals of the reader, stop and resume bit for bit |
| `experiments/`, `ledger.csv`, `TEST_LOG.md`, `DECISIONS.md` | written through `research` (records); none yet in this branch |

## Use (from `autoresearch/`, under WSL)
```
python -m research new toy width "width of the network"        # experiments/E001_width/experiment.md: fill the card
python -m research check toy E1 && python -m research start toy E1
python -m research status toy E1
python -m research judge toy E1 --write                          # verdict.json, ledger.csv, one commit
python -m research adopt toy E1 a --decision D<n>                # the owner's decision: champions/toy/
```
The data (`<data root>/toy/`) and the runs (`<runs root>/toy/`) lie where the machine file says (`research machine`), never in this folder.
Tests: `python tests/test_toy.py` from this folder with `autoresearch/` on `PYTHONPATH` (about a minute; `--quick`: seconds).
