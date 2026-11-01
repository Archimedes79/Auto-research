"""D24 doc edits (exact substring replacements, each must hit once)"""
import os
R = os.environ["BFLC_ROOT"]
def sub(path, old, new):
    p = os.path.join(R, path)
    t = open(p, encoding="utf-8").read()
    assert t.count(old) == 1, (path, old[:60], t.count(old))
    open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))
    print("ok", path)
sub("evaluation/evaluate.py", "Timing of the network (ms per frame on an idle GPU) is not measured here: evaluation/drivers/p17_latency.py (protocol in PROTOCOL.md).",
    "Timing of the network (ms per frame on an idle GPU) is not measured here: evaluation/drivers/p18_latency.py (protocol in PROTOCOL.md).")
sub("evaluation/README.md", "Network timing stays in `drivers/p17_latency.py`.", "Network timing stays in `drivers/p18_latency.py`.")
sub("evaluation/README.md", "`p16_final_eval.py` (builder on a split, threshold sweep, card), `p16_eval.py`, `p16_fast_val.py`",
    "`p16_final_eval.py` (builder on a split, threshold sweep, card), `p16_fast_val.py`")
sub("evaluation/README.md", "`p16_timing.py`, `p17_latency.py`, `p16_pareto.py`", "`p16_timing.py`, `p18_latency.py` (ms per frame of the whole chain: `--union 1|2`, `--fork_prune`, `--rec`), `p16_pareto.py`")
sub("evaluation/README.md", "loop 18: `p18_asm_check.py`, `p18_fork_prune_eval.py`, `p18_regis_eval.py` |", "loop 18: `p18_asm_check.py`, `p18_regis_eval.py` |")
sub("evaluation/README.md", "Move the scripts of a loop here when the loop is filed |",
    "Move the scripts of a loop here when the loop is filed. Moved 2026-10-31 (D24): `loop16/p16_eval.py` (from `drivers/`), `loop16/p16_ensemble.py`, `loop16/p16_hyst.py` (from `training/pipeline/`, no importer), `loop17/p17_latency.py` (superseded by `drivers/p18_latency.py`: same chain, `--union 1 --fork_prune 0` gives the loop-17 setting), `loop18/p18_fork_prune_eval.py` (from `drivers/`) |")
sub("evaluation/PROTOCOL.md", "median of the frames (`p16_timing.py`, `p17_latency.py`);", "median of the frames (`p16_timing.py`, `p18_latency.py`; the loop-17 setting is `--union 1 --fork_prune 0`);")
sub("training/README.md", "`p16_tok_resample.py` (route head asked along the edges), `p16_ensemble.py`, `p16_hyst.py`, `p10_infer.py`",
    "`p16_tok_resample.py` (route head asked along the edges), `p10_infer.py`")
sub("evaluation/drivers/p18_latency.py", "Usage (gcar, GPU, idle machine): p16_pipeline.py --name e123_D7_r3_f1 [--n 60]\"\"\"",
    "Usage (gcar, GPU, idle machine): p18_latency.py --name e123_D7_r3_f1 [--n 60] [--rec 0|1] [--union 1|2] [--fork_prune 0.3]   -> runs/e18/latency/*.json\n(the timing driver of evaluation/PROTOCOL.md since 2026-10-31; p17_latency.py, now in evaluation/loops/loop17/, = --union 1 --fork_prune 0 with output in runs/e17/latency)\"\"\"")
