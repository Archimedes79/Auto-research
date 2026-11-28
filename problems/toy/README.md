# toy - the reference problem of the research system

A disc drifts through a noisy 32 x 32 picture; a small network marks its pixels. Small enough for the CPU in minutes, it goes through data and
ground truth, training, evaluation (numbers, pictures, compute time), a verdict and an adoption. It is the chain test of the core
(`python -m research.selftest`) and the template of a new problem ([`../../autoresearch/EXTENDING.md`](../../autoresearch/EXTENDING.md)).

## What a problem writes
| file | what |
|---|---|
| `__init__.py` (empty, in every code folder) | the problem is the package `toy`: its modules import each other as `from toy.net.model.net import DiscNet`, nothing edits `sys.path` |
| `problem.json` | the steps (prepare, train, evaluate; a step is `{python} -P -m toy.<module>`), the score `seg.iou` with its band, a guard, the control floor, the holdout split, the protected files, the tests |
| `dataset/` | the recipe of the raw data (`make_data.py`) and its manifest (shards with split, size, sha256) |
| `gt/` | the ground-truth generator: `raster.py` is the hashed part; the store is named by the signature (rasteriser, options, data manifest) |
| `net/` | `prep/` the shard reader (checks every shard's hash, guards the holdout), `model/` the network, `loss/` the loss |
| `training/` | `train.py` on `trainkit`, `configs/base.json` (every setting an arm may change), the drawing `architecture.*` |
| `evaluation/` | `adapter.py`: sets, source, subject, a metric with additive counts, the overview picture, the timing |
| `tests/test_toy.py` | the style of the package (`research/pkgcheck.py`, the layers `net` below `training` and `evaluation`), known answers (asymmetric), the refusals of the reader, stop and resume bit for bit |
| `TEST_LOG.md` | the TEST gate of the holdout split; experiments, `ledger.csv` and `DECISIONS.md` come with the first use |

## Use (from `autoresearch/`, under WSL)
```
python3 -m research new toy width "width of the network"      # experiments/E001_width/experiment.md: fill the card
python3 -m research check toy E1 && python3 -m research start toy E1
python3 -m research status toy E1
python3 -m research judge toy E1 --write                      # verdict.json, ledger.csv, one commit
python3 -m research adopt toy E1 a --decision D<n>            # the owner's decision: champions/toy/
```
The data (`<data root>/toy/`) and the runs (`<runs root>/toy/`) lie where the machine file says, never in this folder. Tests:
`python3 -m research tests toy`, or by hand `python3 -m research py toy -m toy.tests.test_toy` (`--quick`: no training). `research py toy ...`
is python with the toy's interpreter and its module path (`problems/`, `autoresearch/`, the repository); nothing is installed.
