import glob, json
p = glob.glob("experiments/E258*/experiment.md")[0]
t = open(p, encoding="utf-8").read()
add = """
## 1b. Amendment after step 0 (2026-10-30, before any variant was measured)
- **Step 0 result** (`profile_step0.txt`, 200 frames, load average 4.2: shares are meaningful, absolute ms are not): of the 4.36 s cumulative, `relocate_forks` 21 %, `anchor_junctions_fast` 16 %, the path-union loop `place` 16 %, `simplify` 11 %, `prune_forks` 6 %. Two stages above 15 % (prediction of the card: hit), so the kill criterion does not fire.
- **Step 1a** (as in the card): change a stage, identical graphs. **Step 1b (added now)**: the two largest stages are post-steps that move vertices to junction-head peaks (`reloc_thr 0.4`, `junc_thr 0.5`); `closed_questions.md` says relocation / anchoring settings add nothing on top of fork pruning (E215, E218), but removing them outright was not tested. Variants on the cached outputs of fast_recall and champion, VAL 873: base, reloc off, anchoring off, both off; Q, J, split / merge F1, topo F1, vert100 and interleaved A/B assembly time (5 repetitions of 200 frames).
- **Rule for 1b**: a stage is dropped from the product (candidate, then the code route with `ar-reviewer`) only if Q loses at most 0.003, J at most 0.01 and vert100 changes by at most 0.3 on BOTH models, and the saving is at least 0.8 ms in relative timing; otherwise the stage stays. Timing is relative, provisional until an idle window.
- **Prediction for 1b**: relocation off saves 1.5 ms (tolerance 0.8) and costs J 0.01 (tolerance 0.01); anchoring off saves 1.0 ms and costs J 0.02.
"""
i = t.index("## 3. Summary")
open(p, "w", encoding="utf-8").write(t[:i] + add.lstrip("\n") + "\n" + t[i:])
