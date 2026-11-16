"""scratch (E343 code step, round 2b): module docstring (provenance, rehearsal command), out_dir of smoke runs per setup"""
import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''def out_dir(setup, split, n=0):
    return os.path.join(dump_dir(split, n), "cards") if n else os.path.join(ROOT, "runs", "eval", f"{setup['out']}_{split}")''',
    '''def out_dir(setup, split, n=0):
    return os.path.join(dump_dir(split, n), "cards_" + setup["out"]) if n else os.path.join(ROOT, "runs", "eval", f"{setup['out']}_{split}")''')

rep('''`all` per seed: the dumps (a dump that exists is VALIDATED and reused, never re-dumped or overwritten; `--b-dump` names B's dump to reuse and it must exist), the scoring of every variant in one pass, then the cards and `results.md` of both seeds.
''', '''`all` per seed: the dumps (a dump that exists is VALIDATED and reused, never re-dumped or overwritten; `--b-dump` names B's dump to reuse and it must exist), the scoring of every variant in one pass, then the cards and `results.md` of both seeds.
Before any dump is made, `all` checks the seeds of the pair runs, the pixel pyramid of the stack, the dumps named by flags and the control card (refused there, not after hours). A dump that this module made (default paths) has a sidecar `<name>_head.json` with
its provenance (run, device, sha256 of the checkpoint, digest of the imported lane code): an existing one is reused only under the same provenance, else refused with a message (nothing is deleted: move it away). Dumps named by a flag (E321's, the stored VAL
dumps) are exempt: validated, and E321's also by hash. Cards and `results.md` are written once (refused when they exist; `--max` runs excepted); they name the device and carry the sha256 of the imported lane code, and `CODE DRIFT: <files>` is printed and
written to `results.md` when one of those files changed during the run (not fatal). A standalone `dump-b` / `dump-ends` never overwrites a dump, and a name without `.pkl` is refused (the head file would overwrite it).
''')

rep('''THE TEST GATE (hard rule 1, D84).''', '''THE DRESS REHEARSAL ON VAL (criterion 2 of the card E343): the TEST command with VAL files, on the CPU, for S = 0 and 1. `verify --seed S` writes `temp/e343/val/val_check/verify_card_e312_p_s<S>.json`, E312's snapped card from the stored dumps in the layout of
E321's `card_s<S>.json`. The plain-graph tolerance of the rehearsal is the loose VAL one by design (a fresh CPU dump against GPU cards); the 1e-9 branch of the TEST kill criterion is covered by a toy test.
  (a) control from the stored VAL files (the TEST layout of the control card, the flagged B dump; E312's counts do NOT reproduce on the fresh B graph, so there is no interval and `reading()` takes its fallback: expected):
      jtest_eval.py dump-b --split val --seed S                                          # B's fresh dump -> temp/e343/val/predfull_e273_p_s<S>.pkl
      jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump runs/e312_p_s{seed}/ends_val.pkl --control-card temp/e343/val/val_check/verify_card_e312_p_s{seed}.json
  (b) control from the SAME fresh dumps (as on TEST, where E321's dumps and card belong together: E312's counts reproduce to 1e-9, the intervals are used):
      jtest_eval.py dump-b --split val --seed S;  jtest_eval.py dump-ends --split val --seed S --run e312_p_s<S>                # temp/e343/val/predfull_e273_p_s<S>.pkl, ends_e312_p_s<S>.pkl
      jtest_eval.py all --split val --run-prefix e312_p_s --seeds 0 1 --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl         # E312's cards of these dumps -> runs/eval/e312_val/card_s<S>.json (E321's layout), checked against its stored cards
      jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json

THE TEST GATE (hard rule 1, D84).''')
open(P, "w", encoding="utf-8").write(s)
print("ok")
