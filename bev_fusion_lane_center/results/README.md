# results/ - the result folders of the lane project

One folder per experiment (`E273/`, `E274/`, ...) or per run (`E273.p_s0/`), and the report pages of an experiment (`E273.report/`, `E276.report/`). A run folder is the result package that `autoresearch/tools/package.py` writes (`package.json`: code tag, GT version, checkpoint, numbers; `provenance.json`: what the number rests on), plus the figures, frames and videos of that run; its own `README.md` says which card it belongs to and the verdict.

An experiment folder holds the `README.md` index of the experiment and its small result files (sweeps, tables). The next card's `results.md` (written by `decide.py --write`, linked from the schedule's `result` column) belongs next to its card in `../experiments/E<id>_<slug>/`; this folder keeps what a run produced.

In git: the `README.md` files, `package.json`, `provenance.json` and small result json files. Not in git: frames, videos and other heavy files (about 1.2 GB on disk); the checkpoints stay in `../runs/`.

Reproduce a run: `python autoresearch/tools/provenance.py repro <run id>`.
