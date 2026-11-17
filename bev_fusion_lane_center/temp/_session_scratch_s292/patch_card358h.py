import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 6. Summary"
assert s.count(a) == 1
txt = ("**Addendum to the pre-statement of attempt (4), 2026-11-16 12:45, still before the run (code critic of the faithful pass: no blocking defect; the code matches the pre-statement; differences it names, none of which changes a number): B's loader here carries `ends=True` / `DEFAULT_ENDS` (B's own dump had neither: only `ent` / `ext` keys are added, and attempt (2) used the same B loader with a matching B side), B's loader uses 2 workers (the dump 3, no numeric effect); the loader flags are put back after the pass; `--bs` with `--faithful` is refused; a REFUSED check prints the largest absolute difference per differing field as a DIAGNOSTIC (the rule stays the tolerance). The critic expects ACCEPT because the B half equals attempt (2) (all frames' `ends` / `attrs` matched) and the pair half equals attempt (3) (`pairs` matched in all 24 frames): the check mainly certifies the join and the record construction. **An ACCEPT does NOT certify the TRAIN pass:** `E358.w_s*` trains on the single-loader live pass (the pair model at batch size 4), which differs from the bs-8 pair dump by up to 0.0022 in one pair of one frame of 24 on seed 0 (the diagnostic); seed 1 has never had a GPU live pass.**" + nl + nl)
s = s.replace(a, txt + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
