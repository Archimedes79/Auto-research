import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''[--control-card P] [--no-stack]   # what run.py launches for a TEST use''',
    '''[--control-card P] [--no-stack] [--expect-code-digest D [--expect-code-card CARD]]   # what run.py launches for a TEST use''')

rep('''its provenance (run, device, sha256 of the checkpoint, digest and file list of the imported lane code): an existing one is reused only under the same provenance, else refused with a message (nothing is deleted: move it away). A dump named by a flag
is exempt when it has no sidecar (E321's, the stored VAL dumps: validated, E321's also by hash) and checked like the others when this module made it.''',
    '''its provenance (run, device, sha256 of the checkpoint, digest and file list of the imported lane code, taken when the dump STARTS): an existing one is reused only under the same provenance, else refused with a message (nothing is deleted: move it away).
A dump named by a flag has no provenance when it is not this module's (E321's pair dump has a `_head.json` without a provenance key, the stored VAL dumps have none): exempt, validated by its rows and, E321's, by its sha256; one with provenance is checked like the others.
Every such refusal comes in the precheck, before a dump of any seed is made. The provenance is taken before the model is loaded; a crash between the dump pickle and its sidecar leaves a dump that looks 'made by other code': move it away.''')

rep('''Cards and `results.md` are written once (refused when they exist; `--max` runs excepted); they name the device and carry the sha256 of the imported lane code, and `CODE DRIFT: <files>` is printed and
written to `results.md` when one of those files changed during the run (not fatal). A standalone `dump-b` / `dump-ends` never overwrites a dump, and a name without `.pkl` is refused (the head file would overwrite it).''',
    '''Cards and `results.md` are written once (refused when they exist; `--max` runs excepted), so before each re-run move the old outputs away (never delete): the cards folder of the earlier run, and the dumps in temp/e343/<split>/ made under another code
digest. They name the device and carry the sha256 of the imported lane code, and `CODE DRIFT: <files>` is printed and written to `results.md` when one of those files changed during the run (not fatal). A standalone `dump-b` / `dump-ends` never overwrites
a dump, and a name without `.pkl` is refused (the head file would overwrite it). The only folders written are temp/e343/, runs/eval/e324s_val, e324s_test and e312_val (an allow-list in `atomic`, next to the deny-list of E321's folders).
The code digest (`code_sha256`: the static import closure of this file over the project's search path, 70 or so files) is printed at the start of every `all`; `--expect-code-digest D` (the `code_digest` of the rehearsal's cards, or its first 16 characters)
refuses the run before any dump when the imported lane code is not the rehearsal's, and `--expect-code-card CARD` (a card of the rehearsal) names the changed files.''')

rep('''THE DRESS REHEARSAL ON VAL (criterion 2 of the card E343): the TEST command with VAL files, on the CPU, for S = 0 and 1.''',
    '''THE DRESS REHEARSAL ON VAL (criterion 2 of the card E343): the TEST command with VAL files, on the CPU, for S = 0 and 1 (outputs of an earlier rehearsal moved away first: see above).''')

rep('''      jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json
''', '''      jtest_eval.py all --split val --test-id E343.v --run-prefix e324_s_s --seeds 0 1 --device cpu --b-dump temp/e343/val/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e343/val/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_val/card_s{seed}.json
  The TEST line is this command with the TEST files of E321 (--b-dump temp/e321/test/predfull_e273_p_s{seed}.pkl --control-ends-dump temp/e321/test/ends_e312_p_s{seed}.pkl --control-card runs/eval/e312_test/card_s{seed}.json), --split test --test-id E343.t,
  and `--expect-code-digest <code_digest of runs/eval/e324s_val/card_s0.json> --expect-code-card runs/eval/e324s_val/card_s0.json`, so that the code of the TEST pass is the code that was rehearsed.
''')
open(P, "w", encoding="utf-8").write(s)
print("ok")
