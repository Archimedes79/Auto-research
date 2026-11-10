# The experiment flow (one experiment, six steps)

Mirko's picture of 2026-11-01: plan card -> coding -> experiment -> training -> evaluation -> store to folder, with the data refreshed where necessary. The lead (or an experiment agent) follows it for every experiment that feeds a verdict; the loop prompt (`loop.md`) is about the rounds around it. Commands run from the problem folder through the WSL (`wsl python3 ../autoresearch/tools/<tool>.py`; bev_fusion: `AR_PROJECT=. ` inside the WSL call).

| # | step | what | tool / file |
|---|---|---|---|
| 1 | **Plan card** | question, prediction (a number), decision rule, arms and control, budget, kill criterion: the card block in section 0 of `experiments/E<id>_<slug>/experiment.md`. A fresh `ar-critic` (mode cards) when the design is new or costs GPU hours | `experiment.py check E<id>`, `templates/experiment.md` |
| 2 | **Coding** | only when the card needs new code: change, spec test, README row, own paths in the commit; shared code goes through an `ar-critic` pass (mode code). A small change is made inline, a larger step goes to `ar-coder` | the folder's selftest, `check_git.py --paths ...` |
| 3 | **Experiment** | (a) the data it needs exist: the nuScenes set named in the config (small or full, with its split) and the GT store of the GT version the run will use (`data/autogt/<signature>/index.json`); a missing store is built first with the `GtGenerator` of that version, as its own item of the schedule (a full build takes hours); (b) a planned ledger row per arm and seed; (c) start through the launcher, which commits the code and tags it `bf/<id>` / `lcr/<id>` | `ledger.py add`, `run.py --id ... --hyp ... --pred ... --rule ... --require-card -- <command>`, `GtGenerator/AGENTS.md` |
| 4 | **Training** | weights and the graph come out; watched by code (kill rules), the lead is woken by an alarm, nobody polls | `Training status / watch`, `schedule.py show` |
| 5 | **Evaluation** | integrate the network into the evaluation, run it on the frozen split, text results and a card with its provenance block; the verdict by the rule of the card | lane `evaluate.py <registry name>`, bev_fusion `bev_suite.py eval`, `decide.py --experiment E<id> --write` |
| 6 | **Store to folder** | the result package `results/<run id>/`: the card, the code tag, the data reference (raw-data manifest, split, frame hash), the GT version, the evaluation version, the numbers as text and the list of the heavy files (pictures, videos) with how to rebuild them; the ledger note; the best three are updated | `provenance.py manifest <run id>`, `package.py package <run id>`, `ledger.py note`, `keep3.py consider` |

After the experiment: derived data of runs outside the best three and GT stores older than the latest are cleaned by `package.py clean` and `package.py gt` (a dry run lists what would go; `--apply` deletes, D40). Reproduce a run later: `provenance.py repro <run id>`.

Defaults, not hard rules (except what `METHOD.md` lists as hard: TEST only under a recorded OK, a verdict from a rule written before the run). Do the sensible thing when a step does not fit a case, and write down what you did differently and why.
