# 8. Simplification: what was removed

**In short.** Simple is a goal of the project (D113): one head, few steps, little code that has to be understood. This chapter counts what the clean-up of 2026-11-18 removed. The table fills itself from git: `tools/simplify_stat.py` reads the files and lines of named path groups at a BEFORE ref (an archive tag made before the removal) and at the AFTER ref (the landed tree) and writes the numbers that this chapter reads (the `values` of the report). Nothing here is typed by hand; a cell that the tool did not fill is a PENDING mark.

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
| tools and supervisors | {{simp_tools_files_before}} | {{simp_tools_files_after}} | {{simp_tools_lines_before}} | {{simp_tools_lines_after}} | {{simp_tools_lines_removed}} | {{simp_tools_files_removed}} |

Table: What the removal did to the system (parameters and time come from the cards, not from git).

| quantity | before | after | source |
|---|---|---|---|
| parameters of the delivered path | 2.32 M (B' 2.196 M + head 0.127 M) | PENDING (unchanged expected: the blocks were not in the champion) | token cards, `head_params` |
| ms per frame of the whole path | 16.83 / 16.57 | PENDING: measured in the quiet gap | E365 `timing`, chapter 7 |
| the champion's numbers (topo F1 s0 / s1) | 0.6679 / 0.6762 | PENDING: the same frames, the same weights, re-read on the landed tree | the card of the re-check |

**How to rebuild this table.** `python3 tools/simplify_stat.py --before <archive tag> --after HEAD --group token=<pathspecs> ... --out parts/figures/simplification.json`; the report's `values` point into that file. The tool counts what git tracks, never the working tree.
