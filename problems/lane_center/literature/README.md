# literature/ - the lane project's papers

Taken over from the old system (git tag `v1`) on 2026-11-27. New research runs as a topic, by the procedure of
[autoresearch/literature](../../../autoresearch/literature/README.md): one folder per topic under `topics/`, one note per paper, a generated
table, a spot check, an overview.

Two files to read, one table to edit:

- **[CATALOGUE.md](CATALOGUE.md)** - the one list of every paper of this folder (generated): what it is, link to the original and to its
  code, where it lives here (note, deep extraction, code extraction), and why it is relevant for our problem or why not. Relevant papers
  first. For an agent: grep an id or an axis (A1 - A9).
- **[OVERVIEW.md](OVERVIEW.md)** - what we look for and why, the nine axes with their core papers, the open gaps, the rules of this folder.
- **[papers.csv](papers.csv)** - the only table to edit: per paper the axis, tier, verdict (use / watch / context / not-relevant), reason
  code, statement, extraction plan, code and licence overrides. A paper a topic found that is worth keeping gets a row here (id: its arXiv
  number or a short id; title, url, idea). After a change, in this folder: `python3 catalogue.py all`.

| folder / file | content |
|---|---|
| `topics/` | research topics of the current procedure: candidates, one subfolder per paper (`note.md` and extractions), `check.md`, `overview.md` |
| `papers/` | level 1 of the corpus: one note per paper family (method level, from page digests); `<ID>_<slug>.md` |
| `deep/` | level 2: full-text extraction of core papers; rules and template in its README |
| `code/` | level 3: code extraction per repository, with a port card; the process and the template in its README. The `<repo>.md` files are not in git |
| `register.md` | technique register: K ids with source, status, where used |
| `catalogue.py` | builds and checks CATALOGUE.md and the generated blocks of OVERVIEW.md (`python3 catalogue.py selftest`) |

The code extractions that were made before 2026-11-27 are saved outside git in the copy of the old folder (data archive,
`files/bev_fusion_lane_center/literature/code/`); copy them into `code/` on a machine where their links should work.

## Format of the corpus notes (`papers/`)
The notes were written as cards; these are the values of their fields.

## Stage taxonomy (field `stage`)
S1 seeding / initialisation of hypotheses · S2 sampling pattern and feature access · S3 verification / scoring along a line or around a point · S4 hypothesis generation and selection (children, beams, NMS, top-k) · S5 refinement / cascade / iterative update · S6 topology and linking (adjacency, link head) · S7 graph assembly / merge / DAG conversion · S8 temporal propagation and fusion · S9 route / SD-map / navigation attribution · S10 planner interface · L loss and label construction · T training regime · E evaluation metric / protocol · D data and labels · B BEV encoder / sensor fusion · X other.

## Vocabulary for the compatibility fields
- **representation** (input / output): `bev_raster`, `image_feat`, `queries`, `point_set`, `polyline`, `bezier`, `segments`, `graph(nodes+edges)`, `adjacency`, `token_sequence`, `scores`, `map_raster`.
- **regime**: `teacher_forcing`, `dagger`, `free_rollout`, `rl`, `next_token`, `hungarian_set`, `matching_free`, `wta`, `denoising`, `self_supervised`, `none`.
- **differentiable**: `yes` (gradient flows through the block and its selections), `partial` (breaks at a named operation), `no`. Name the breaking operation: argmax, sampling, hungarian, nms, topk, tokenisation, rasterisation, rule.
- **sensors / data**: `camera`, `lidar`, `radar`, `aerial`, `any_bev`; dataset names; BEV range / resolution if stated.
- **fit_to_us**: `reuse_idea` (take the principle), `reimplement` (small, from the method section), `adapt_code` (their code can be ported; name the blocker), `baseline` (use as comparison), `not_suitable` (say why). Effort `S` (< 1 day), `M` (days), `L` (week or more).
- **compatibility flags between two techniques** (the synthesis derives them from the card fields; write the reasons into `compat_needs` / `compat_conflicts` so they can): `direct`, `adapter` (name it, e.g. bezier <-> polyline resampling, query <-> bilinear sample), `conflict` (reason: differentiability / regime / sensor / interface / compute), `unknown`.

## Card format (strict, one block per technique; keep each under about 25 lines)
```
### CARD <PREFIX><nn> - <technique name>
- papers: <title> (<authors short>, <venue year>) <link>; ... (all papers the card rests on)
- depth: A | M | C | M+C
- stage: <one or two of S1..S10, L, T, E, D, B, X>
- idea: <2 - 4 sentences, own words>
- formulation: <key equations / pseudo-code in plain text and the hyper-parameters with values; "not read" if depth A>
- input: <representation and shapes if stated>
- output: <representation>
- training: <loss, labels needed, regime>
- differentiable: yes | partial | no - <where it breaks>
- sensors_data: <sensor assumptions, datasets, BEV range and resolution>
- results: <numbers seen, with dataset + metric + source; or "none seen">
- compute: <parameters / latency if stated, else "not stated">
- code: <repo URL or "none found" or "not checked"> (licence and framework are collected by the cluster EV; give the URL only)
- fit_to_us: <reuse_idea | reimplement | adapt_code | baseline | not_suitable> - effort <S|M|L> - <main obstacle in one sentence>
- compat_needs: <what it requires of neighbouring blocks, e.g. "needs a query set with reference points", "needs Hungarian matching", "needs a sequence order of the vertices">
- compat_conflicts: <what it cannot be combined with and why>
- tried_before: <project experiment id + verdict, or "none">
- verified: checked | partly | from memory
```
