import io, sys
sys.path.insert(0, "../autoresearch/tools")
import testgate as TG
row = ("| 2026-11-15 | 23 | the system of record E324 s (the E320 recipe at 6000 pair steps with the geometry loss and the attachment, on the E307 maps; read-outs: snapped = the card of record of D94, snap_nogeom, PRIMARY = attached ends + snap_nogeom, and the stack E338 = the same pair head on the delivered graph of c2) against E312 (E321's stored TEST dumps and card, CPU dumps) as the control, seeds 0 and 1, TEST 833 frames, 21 scenes; the pair head of E324 s and c2's B are dumped on the GPU, B's E273 dump is E321's CPU dump, the scoring is CPU "
       "| run id E343.t (`evaluation/jtest_eval.py all --split test --test-id E343.t --run-prefix e324_s_s --device cuda`, card `experiments/E343_test-e324s/experiment.md`; under run.py with a GPU slot for the dumps, after the VAL dress rehearsal of the same code: the CPU branch exceeded the tolerance of the card, the GPU branch of the card was rehearsed) "
       "| - "
       "| Mirko's OK, once (D95, 2026-11-14, \"2. ja\"; the three readings of the card confirmed 2026-11-14 about 22:25 with \"mach alles\" and again about 23:00 with \"ne, du hast mein ok\"); E324 s with E312 as the control and, for arm k, c2 and the stack E338; report only, no promotion of p or k, nothing selected or tuned on TEST |")
text = io.open("evaluation/TEST_USAGE_LOG.md", encoding="utf-8").read()
lines = text.split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("|"))
out = "\n".join(lines[:last + 1] + [row] + lines[last + 1:])
print("refusal:", repr(TG.test_log_refusal(out, "E343.t", frozenset())))
io.open("temp/_session_scratch_s292/e343_row.txt", "w", encoding="utf-8").write(row)
print(len(row))
