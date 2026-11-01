# results/ - the result package of each run

One folder per run id (`results/<run id>/`), written by `autoresearch/tools/package.py` and `autoresearch/tools/provenance.py` (D40). It answers: how was this run made, what came out, and where is the deeper material.

| file | what | in git |
|---|---|---|
| `package.json` | the package: code tag and commit, GT version and signature, model bundle hash, checkpoint and step, seed, run status, tier (best three or normal), the top-level numbers of `eval/metrics.json` and `eval/latency.json`, and the list of DERIVED files (pictures, videos, pickles, per-scene folders; name and size) | yes |
| `provenance.json` | the full provenance block (`provenance.py manifest <run id>`): GT it trained on, final checkpoint hash, config hash, environment of the run record, the model bundle (the network and loss files with their hash) | yes |
| `bundle/` | a copy of the network and loss files and the config of the run, so the model can be rebuilt from this folder and the checkpoint; the tag holds the same code | no |

The heavy files stay in `runs/<run folder>/` (checkpoints, `eval/`, `report/`). Outside the best three (`records/keep3.json`) the derived files are deleted by `package.py clean --apply` when the run is final, its tag and checkpoint exist and the files are older than two days; the package then says `deleted` with the list and how to rebuild them. Old GT stores go the same way (`package.py gt`). Checkpoints are never removed by it (best three complete, the others slimmed or archived). Every deletion is logged in `records/CLEAN_LOG.md`.

Reproduce a run: `python autoresearch/tools/provenance.py repro <run id>`.
