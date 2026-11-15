import io
p = "experiments/E343_test-e324s/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:80], s.count(a))
    s = s.replace(a, b)
rep("**Status 2026-11-14 (code step done by a coder, VAL only; the code critic pass is running; criterion 2 is not run yet):**",
    "**Status 2026-11-15 05:15 (code built in four coder rounds and three `ar-critic` code passes: round 1 - 2 REVISE, round 3 REVISE small (11 findings, all answered in round 4), the delta check of round 4 ACCEPT; 37 tests pass, about 140 mutants of the rules caught; VAL only; criterion 2, the dress rehearsal, is next: chain `temp/day26/day26.sh`; the log row and the grant come after it and only after it passed):**")
rep("the record of the sha256 of the imported code files before and after the run is NOT implemented (the launcher notes code drift as in E321)",
    "the sha256 record of the imported code IS implemented in round 3 / 4 (`code_sha256`: a digest over 71 files, printed at the start of `all`, stored in every card, drift printed and written to `results.md`) and is the PIN of the TEST line (`--expect-code-card`, below)")
rep("\n     --b-dump temp/e321/test/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e321/test/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_test/card_s{seed}.json\n```",
    "\n     --boot 1000 --b-dump temp/e321/test/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e321/test/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_test/card_s{seed}.json \\n     --expect-code-card runs/eval/e324s_val/card_s0.json\n```")
rep("**`--no-stack` is the ONLY permitted change of this line:**",
    "**The line is final (2026-11-15, after the code critic): it carries `--boot 1000` and the code pin `--expect-code-card runs/eval/e324s_val/card_s0.json` (the card of the passed VAL rehearsal: the digest of the 71 lane code files and the environment, i.e. the hashes of `splits.json`, `val_tokens.json`, `poses_val.npz`, the runs' configs and the versions of torch, numpy and shapely, are taken from it; `--expect-code-digest <its code_digest>` may be added as an extra check; the code refuses the TEST run when the card is not a full, passed, undisturbed VAL rehearsal of this system on the same device, or the code or the environment differ, before any dump). These flags change nothing that is computed. `--no-stack` remains the ONLY permitted change of the computation:**")
rep("No commit or edit of the imported files from the final critic pass until the run ends is agreed with the lane lead,",
    "No commit or edit of the imported files (the 71 of the digest, among them `jwire.py`, `jwire_head.py`, `jends.py`, `qends_eval.py`, `jends_eval.py`, `jtest_eval.py`) from the start of the rehearsal until the TEST run ends is agreed with the lane lead (any edit invalidates the rehearsal: the TEST run refuses; E350's phase B waits for this reason),")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
