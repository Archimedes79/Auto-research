# sources/ - provenance: the original research notes of the loops

These files are what the 404 paper notes, `register.md` and `repos/README.md` were built from. They are kept unchanged as the record of what was read when, by which method (abstract page, HTML full text, search result) and what each work gave in our experiments. Work from [../OVERVIEW.md](../OVERVIEW.md) and [../CATALOGUE.md](../CATALOGUE.md); open these only to trace a claim back or to read the loop-by-loop story of how a paper entered an experiment. Numbers in them are author-reported on mixed protocols.

| file(s) | date | scope |
|---|---|---|
| `lit7/lit7_a.md`, `lit7_b.md` | 2026-10-15 | scan of online lane-graph / centre-line mapping, SD-map priors, temporal fusion, robustness (2024 - mid 2026); abstracts checked |
| `lit8/lit8_a.md`, `lit8_b.md` | 2026-10-16 | pixel-to-sparse-graph conversion, calibrated abstention, planner interfaces from a lane graph |
| `lit_extra.md` | loop 6 | fine-tuning and weight-space methods (LP-FT, WiSE-FT, soups, L2-SP) |
| `lit_extra8.md` | loop 8 | perception-first lane graphs, what planners receive |
| `lit_extra10.md`, `lit_extra10b.md` | loop 10 | from algorithm to learned network; deep learning as a catalogue of building blocks |
| `lit_extra11.md` | loop 11 | temporal fusion as a block of the learned chain |
| `lit_extra12.md` | loop 12 | calibrated outputs and failure monitors at the planner hand-over |
| `lit_extra13.md` | loop 13 | verification blocks for graph hypotheses, trust and ego state |
| `lit_extra14.md`, `lit_extra14r.md` | loop 14 | 14 verified entries 2025 - mid 2026; planner interface, failure training, route attribute |
| `lit_extra15.md` | 2026-10-23 | differentiable iterative point generation for online maps and lane graphs; hand-over to a planner |
| `lit_p16_a_sparsification.md` | 2026-10-23 | sparsification: dense field or many candidates -> few correct vertices and edges |
| `lit_p16_links_a.md`, `lit_p16_links_b.md` | 2026-10-23 | link and venue checks of the p16 literature |
| `lit_p16_tree.md` | 2026-10-23 | verified-ray hypothesis tree and spray-and-link |
| `proposal_sparse.md` | loop 10 | proposal text for sparse graph extraction (element-level mapping of the literature) |
| `lit16/` | 2026-10-23 | the seven loop-16 clusters (growth, topology, vectormap_temporal, multihyp, verify_sampling, planner_nav, sensors_bev), evaluation / data / code inventory `eval_data_code.md`, `cards.csv` (301 cards), `SCHEMA.md` (note fields), `PROJECT_CONTEXT.md` (the project as the research agents saw it) |
| `lit17/` | 2026-10-24 | loop-17 clusters A streaming memory, B failure robustness, C map matching and planner (103 notes, read through page digests; the caveat is in section 1 of each file) |
