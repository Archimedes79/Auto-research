# deep/ - level 2: full-text extraction of the core papers

One file per core paper (tier 1 in [../CATALOGUE.md](../CATALOGUE.md)), named `<ID>_<slug>.md` after its home note. The level-1 note in `../papers/` was written from page digests and says what the paper is; the file here says how to rebuild it: architecture with shapes, equations, loss weights, label construction, training recipe, inference rules, numbers with table numbers, differences between paper and code, and what transfers to our frozen-BEV pipeline. Template: [_TEMPLATE.md](_TEMPLATE.md). Overview and goal: [../OVERVIEW.md](../OVERVIEW.md).

## Rules for the agent that extracts

1. Take a core paper (tier 1) without a deep file; the generated line "Core papers" in [../OVERVIEW.md](../OVERVIEW.md) section 4 names the queue (none left on 2026-10-25); one paper per agent; do not start a paper that already has a file.
2. Reading pages is fine: arXiv abstract and **HTML** version (`arxiv.org/html/<id>`, `ar5iv`), project pages, OpenReview, GitHub pages. The reader often returns PDFs as binary: prefer HTML, then the appendix / supplement page.
3. **No download, no git clone, no pip without the user's OK.** Ask first and give file name, source and size.
4. Mark every section **checked** (page seen), **digest** (seen through a summarising tool, numbers may be mis-transcribed) or **memory** (not seen). Never write a number you did not see.
5. Paraphrase; keep equations and numbers as facts; at most one short quote per paper. Numbers of different papers are on different protocols: do not compare them with each other or with ours.
6. Code of the same paper goes to `../code/<repo>.md` (template there). GPL-3, AGPL and repositories without a licence file are **read only**: describe, never copy.
7. A new technique gets a K id in `../register.md` before an experiment uses it. If the extraction changes why a paper matters, edit its statement in `../papers.csv`.
8. Finish with `wsl python3 tools/literature/catalogue.py all` (links and ids are checked) and report the file path.
