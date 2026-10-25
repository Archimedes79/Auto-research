# literature/ - folder map

Two files to read, one table to edit:

- **[CATALOGUE.md](CATALOGUE.md)** - the one reference for every paper (generated, 456 rows): what it is, link to the original, where it lives here (note, deep extraction, code extraction), what it contains, and why it is relevant for our problem or why not. Relevant papers first, probably-not-relevant last (by reason), code extractions at the end. For an agent: grep the id or an axis (A1 - A9).
- **[OVERVIEW.md](OVERVIEW.md)** - what we look for and why (navigate anywhere with an SD-map route and no HD map: lane centre-lines dense then sparse, temporal fusion of the detections, assignment to a map graph at lane level with road level as fallback, navigation context for the planner), the nine axes with their core papers, what the extractions found, the open gaps, the rules of this folder.
- **[papers.csv](papers.csv)** - the only table to edit: per paper the axis, tier, verdict (use / watch / context / not-relevant), reason code, 3 - 4 sentence statement, extraction plan, code and licence overrides. Papers found by a search and not yet read are rows with their own title / url / idea. After a change: `wsl python3 tools/literature/catalogue.py all`.

Detail behind the two files:

| folder / file | content |
|---|---|
| `papers/` | level 1: 404 notes, one per paper family (method level, from page digests); `<ID>_<slug>.md` |
| `deep/` | level 2: full-text extraction of core papers; rules and template in its README |
| `code/` | level 3: code extraction per repository, with a port card; the process (select, pull, make readable, summarise, link) and template in its README |
| `register.md` | technique register: K ids with source, status, where used (a K id comes before an experiment uses a technique) |

In git: everything above except the per-repository files `code/<repo>.md` (third-party code is extracted on the fly by the process in `code/README.md`; those files stay on disk but are not stored, so their links in CATALOGUE.md work locally only). Rules: `.gitignore` of the repository.

Not here any more: the original research notes of the loops (`experiment_execution/history/literature_sources/`) and superseded files (`_to_delete/literature_lean_2026-10-25/`).
