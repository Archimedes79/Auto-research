# <repo> - code extraction

Copy this file to `code/<repo>.md` (repository name, lower case; the catalogue links it by that name). Fill every section; "not found" is an answer. English. Delete these instructions.

| field | value |
|---|---|
| repository | URL, owner / name |
| papers covered | links to `../deep/<ID>_*.md` and `../papers/<ID>_*.md` |
| state read | commit hash and date, branch; how it was read (GitHub pages / raw files / local clone) - **no clone or download without the user's OK** |
| licence | text of the LICENSE file as seen (name + version), data / checkpoint licence separately |
| stack | Python, PyTorch, CUDA, mm* packages, custom ops, other heavy dependencies |
| size, checkpoints, data | lines of code that matter, availability of weights and data, download size if stated |
| extracted by, date | agent / session, YYYY-MM-DD |

## Port card
The machine-readable summary; `tools/literature/catalogue.py` builds the code table at the end of `../CATALOGUE.md` from it. One line per field, values from the lists, nothing else in the second column.

| field | value |
|---|---|
| repo | URL |
| commit | short hash and date of the state read, or "web pages only" |
| licence | SPDX-like name, e.g. MIT, Apache-2.0, GPL-3.0, NVIDIA-NC, CC-BY-NC, none |
| portable | pure-torch / numpy / mmcv / cuda-op / mixed |
| reuse | read-only / adapt / vendored / metric-only / idea-only |
| effort | S (hours) / M (days) / L (weeks) / none (nothing planned); for the whole planned port, the largest item decides |
| port_targets | our files to create or extend (e.g. `evaluation/openlane/metrics.py`), or none |
| status | read / ported / verified (verified = parity test against the reference output passed) |
| papers | note ids, e.g. PL01 C01 |

## 1. Map of the code
Start from the generated skeleton `<repo>.skeleton.md` if a checkout exists (`python3 tools/literature/code_digest.py <checkout> --paths <sub-paths> --out literature/code/<repo>.skeleton.md`); it lists files, outlines and a keyword index, no bodies. Write the narrative here.
| path | what it does | lines | portable to our env? |
|---|---|---|---|
| `models/xyz.py` | e.g. decoder | 320 | pure PyTorch / needs mmcv / custom CUDA op |

Portable = runs on PyTorch 2.8, Python 3.12, no mmcv / mmdet / mmdet3d (our env `gcar`).

## 2. Key algorithms (where, how)
For each part we may want: file and function / class name with line, shapes in and out, the logic in your own words, hyper-parameters used by the default config. Typical parts: label construction, assignment / matching, loss terms, sampling or attention core, graph assembly / merge rules, temporal propagation, metric implementation.

## 3. Config versus paper
Differences between default config, README and paper text.

## 4. Data and label pipeline
How ground truth is generated (nuScenes map expansion, lane connectors, resampling, direction), what augmentations run, what the data loader expects. Often the most reusable part.

## 5. Evaluation code
Metric definitions as implemented (matching thresholds, resampling, tolerances, devkit version), so our numbers can be mapped.

## 6. Port plan
| item | decision |
|---|---|
| licence consequence | permissive (MIT / Apache / BSD): adapt with attribution and keep the notice. GPL-3 / AGPL / no licence file: **read only**, re-implement from the description in your own words, copy no code |
| what to lift | list: function -> where it would live in `training/` or `evaluation/` |
| dependency gaps | mmcv / mmdet ops, custom CUDA, old PyTorch APIs -> replacement (e.g. pure-PyTorch deformable core) |
| effort, risk | S / M / L and the blocker |

## 7. Gotchas
Bugs, silent defaults, non-determinism, coordinate conventions (x forward? y left? metres or pixels? origin), things that cost time.

## 8. Snippets
Permissive licences only, at most 15 lines each, with file and line. For GPL / unlicensed code: none.
