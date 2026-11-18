# 8. Simplification: what was removed

**In short.** Simple is a goal of the project (D113): one head, few steps, little code that has to be understood. This chapter counts what the clean-up of 2026-11-18 removed. The table fills itself from git: `tools/simplify_stat.py` reads the files and lines of named path groups at a BEFORE ref (an archive tag made before the removal) and at the AFTER ref (the landed tree) and writes the numbers that this chapter reads (the `values` of the report). The line counts are never typed by hand; the three rows of the second table come from cards.

**What was removed, and why each part was safe to remove (decisions of the lead, D117 / D119).**
- The S2 (past) and S3 (map) blocks of the token head: both are a TIE against the champion (chapter 6), so the cheaper variant is none; their code, trainers and read-outs go, behind an archive tag.
- The jtok2 and the recursive-state families of earlier experiments (E354 and its helpers), except the small timing helper the read-outs still use.
- In the evaluator: the fixture, the controls, the parity check and the default sweep grid (the per-class grid stays).
- The supervisor scripts of single experiments: one arm tool (`jarm.py`) replaces them.

Table: Files and lines of the code groups, before and after (git, `simplify_stat.py`; the refs are the archive tags and the landed commit).

| group | files before | files after | lines before | lines after | lines removed | files removed |
|---|---|---|---|---|---|---|
| token head (model, trainer, loader) | {{simp_token_files_before}} | {{simp_token_files_after}} | {{simp_token_lines_before}} | {{simp_token_lines_after}} | {{simp_token_lines_removed}} | {{simp_token_files_removed}} |
| S2 / S3 blocks and read-outs | {{simp_blocks_files_before}} | {{simp_blocks_files_after}} | {{simp_blocks_lines_before}} | {{simp_blocks_lines_after}} | {{simp_blocks_lines_removed}} | {{simp_blocks_files_removed}} |
| evaluator (`jtoken_eval.py` and its tests) | {{simp_eval_files_before}} | {{simp_eval_files_after}} | {{simp_eval_lines_before}} | {{simp_eval_lines_after}} | {{simp_eval_lines_removed}} | {{simp_eval_files_removed}} |
| arm tool and supervisors (`jarm.py` replaces the per-experiment scripts; the keep-alive supervisor and its test left) | {{simp_tools_files_before}} | {{simp_tools_files_after}} | {{simp_tools_lines_before}} | {{simp_tools_lines_after}} | {{simp_tools_lines_removed}} | {{simp_tools_files_removed}} |

Table: What the removal did to the system (parameters and time come from the cards, not from git).

| quantity | before | after | source |
|---|---|---|---|
| parameters of the delivered path | 2.32 M (B' 2.196 M + head 0.127 M) | unchanged: the removed blocks were not in the champion path (not re-counted after the landing) | token cards, `head_params`, E2E cost table |
| ms per frame of the whole path | 16.83 / 16.57 (E365, s0 / s1); 16.21 (quiet re-measurement of the night, before the landing) | not re-measured after the landing: the modules of the inference path are untouched and the live read below is bit for bit; 16.21 stays the number | E365 `timing`, E2E cost table, chapter 7 |
| the champion's numbers (D120 champion, seed 0, topo F1 / junction F1) | 0.6682 / 0.2538 (stored live card) | 0.6682 / 0.2538: B' live on the GPU + the head + the decode + the scorer on the landed tree, 873 VAL frames, 81 metrics, 0 differing (bit for bit); seed 1 not re-read | E385 read-out log (`E372.repro_s4_s0`) |

**Reading the table.** The BEFORE ref is the archive tag `archive/jtoken-before-simplify-2026-11-17` (the tree before the clean-up), the AFTER ref is the committed tree of the build. The "tools" row grows by 23 lines: the one arm tool `jarm.py` is larger than the keep-alive supervisor it replaces, and it carries the card words, the curve check and the lock for every arm. The speed path of the token-head training (E383) was added and removed on the same day: it is not in the BEFORE tree and not in the AFTER tree, so it shows as zero here; the code is kept behind the local tag `archive/token-speed-path-2026-11-18`, and what stayed from it is `pmap.py`, the `--jobs` option of the sweep and one relative-root fix of the token loader. The S2b experiment (E387, chapter 6) restored the S2 files for its runs and removed them again after the TIE (the restored copies were never committed to main; the S2 and S3 files of E368 / E374 had been on main and are counted as removed above), and the hook for the past block in `jtoken.py` was on main from the commit `0a7e85c1` to `2617b99927` (default off, no number changed) and is gone; the token group above is counted on the tree after that.

**How to rebuild this table.** `python3 tools/simplify_stat.py --before <archive tag> --after HEAD --group token=<pathspecs> ... --out parts/figures/simplification.json`; the report's `values` point into that file. The tool counts what git tracks, never the working tree.
