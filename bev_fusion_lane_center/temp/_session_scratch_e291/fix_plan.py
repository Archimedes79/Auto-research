p = "temp/e291/install_plan.md"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
sub("`run_staged_gpu.sh` REFUSES (exit 2) when the stage has no `jends.py`: a twin on the real code would be no twin.", "`run_staged_gpu.sh` REFUSES (exit 2) when the stage has no `jends.py` (a twin on the real code would be no twin) and, outside the tests (`run_staged.sh` sets `E291_TEST=1`), when `BFLC_STAGE` is anything but `temp/_session_scratch_e291/stage`: E347's wrappers use the same variable name, and a shell that exports E347's stage must not start an arm.\n`start_arms.sh` sets `BFLC_STAGE` unconditionally to this stage and, before every start, (1) checks the freeze: sha256 of the four staged files, `training/grid/jprior.py`, `evaluation/jdiag.py` and `temp/e347/stage_launch.py` against `temp/e291/stage_sha256.txt` (REFUSED, exit 3, on a mismatch or a missing list: after a deliberate change of one of them write the list again: `sha256sum <the seven paths> > temp/e291/stage_sha256.txt` from the problem folder), (2) refuses (exit 4) an existing `runs/e291_<arm>_s<S>` unless `E291_RESUME=1` (a stopped run resumed by hand; `run.py`'s own retries do not pass through the script and resume from `ckpt.pt`). `E291_DRY=1` prints the `run.py` line instead of starting it. Tested by `temp/e291/test_e291_guards.py` (plain python, 3 s).")
sub("--prior-mode clean,noise10,noise30,pm1,u,s        # arm b (6 folders runs/e291_b_s<S>_<readout>)", "--prior-mode clean,noise10,noise30,pm1,u,s,sc     # arm b (7 folders runs/e291_b_s<S>_<readout>)")
sub("--prior-mode clean,noise10,noise30,pm1,s          # arm a (U is read for b only)", "--prior-mode clean,noise10,noise30,pm1,s,sc       # arm a (U is read for b only)")
sub("<R> in clean noise10 noise30 pm1 u s", "<R> in clean noise10 noise30 pm1 u s sc")
sub("# kill (5): S - twin", "# kill (5): S - twin (and the same with e291_b_s<S>_sc / ends_card_sc_snap_thr0.1: the kill reads both s and sc)")
s += '''
## 7. Fix round (code critic REVISE, 2026-11-15)

- **The `sc` read-out** (`Spec.readout("SC")`, folder `e291_<arm>_s<S>_sc`): the S table put the partner's arms in the partner's ego frame, so most keypoints see no arm and S is close to "no map". SC keeps the frame's OWN arm centres and axes (the clean table) and takes the counts from the partner frame (the same partner as S): arm i gets the count of the partner's arm i modulo the partner's number of arms (rows in the order of the clean tables, both directions); a partner without arms gives unknown (both directions). It differs from the clean table in the counts only (tested: deterministic, own `c` / `t` kept, differs on most frames with arms, the keypoints see the same arms as in the clean and U read-outs). The dump prints, per read-out, the share of the valid keypoints that have an arm of the table in reach (`ends_head.json` -> `prior.arm_share`): the check that S is nearly an empty table and SC is not. The kill (5) of the card reads both s and sc; the dump of the twin is unchanged.
- **Dump folders are atomic**: every name (`<run>_<readout>`, with `_max<N>` for a `--max` test, which therefore never blocks the real names) is checked first, the folders are made when the pass is done: a crash or an existing name leaves nothing behind (tested with a crashing loader and a half-existing list).
- **U** zeroes `n / 4`, the rank term and `(found - n) / 4` as well (the geometry features stay): the C clause credits the count together with the within-arm rank and found terms (the card is reworded by the lead; no code change).
- **For the E347 and E350 chats (after the install):** the BFLC_SIG assert of `jends_train.py` stops every launch whose `BFLC_SIG` does not match the `sigma` of the maps checkpoint (a maps checkpoint without `sigma` counts as 1.5): their launch scripts must export it (E347's 2 m maps trained with `--sigma 1.0`: `BFLC_SIG=1.0`).
- **Ledger rows**: a retried arm (`run.py --retries`) resumes from `ckpt.pt` with another data order than an uninterrupted run: write "resumed from step N" into its ledger row.
- **README rows** are corrected: `training/train/README.md` and `training/data/README.md` do not exist; the `jends_train.py` sentence goes into the `train/` row of `training/README.md`, the `lane_gt_data.py` sentence into the new `jprior.py` row of `training/grid/README.md`; the promoted test needs neither `ref` nor `identity_real` (the latter skips without `ref_e312.pt`).
- `partner_arms` keeps the last partner (S and SC of one frame load the partner's GT once).
'''
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
