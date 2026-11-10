# jobs/running - the Runner (`bev_fusion/runner`, class `Runner`)

The Runner takes arguments, not job files: `python -m runner Runner.plan runs/t23_C_s1 small.val` looks, `python -m runner Runner.predict runs/t23_C_s1 small.val --taps fused` runs and stores (from the
`bev_fusion` folder, inside wsl). Everything about it, the commands, which networks run and the memory rule (D70), is in `../../../runner/README.md`.

A run that feeds a verdict goes through `autoresearch/tools/run.py` with that command as its command; `predict` called directly is a diagnostic (no ledger row, no GPU slot).
