# Schema of the literature research "compatibility space" (all agents follow this)

Goal: capture, for every relevant technique, **what it is, how exactly it works, what it needs and what it excludes**, so that techniques can later be mixed systematically and tested in our pipeline (see `PROJECT_CONTEXT.md`). Depth matters more than breadth: a card read from the method section is worth more than ten abstract-level cards.

## Rules
1. Read `PROJECT_CONTEXT.md` first. Judge fit against OUR interfaces (cached 200 x 200 BEV feature pyramid, ego metres, vertex + token + validity + edges output, matching-free losses, WSL env without mmcv / mmdet, RTX 3060).
2. **Reading only.** Use WebSearch and WebFetch. Do not download files, do not use git, curl, pip or any package manager, do not write anything to disk except your own output files named below. Prefer arXiv HTML pages (`arxiv.org/html/<id>`, `ar5iv.labs.arxiv.org/html/<id>`), abs pages, OpenReview / CVF / proceedings pages, GitHub README and LICENSE pages as web pages. If only a PDF exists, fetching its text via WebFetch is allowed.
3. **Depth code per card:** `A` = abstract or listing page only; `M` = method section read (equations, architecture figure description, loss, training details); `C` = code page read (README, licence, framework statement). Combine, e.g. `M+C`. Never claim a depth you did not reach. Facts not seen on a page must be written "(from memory)" or "not found".
4. **Numbers:** only numbers you saw, with dataset, metric and where (paper table / abstract). If a number differs between versions say so. Never invent numbers, hyper-parameters or equations; write "not stated" instead.
5. **Own words.** No long quotes (max one quote under 15 words per source). No copying of tables beyond a few numbers. Never reproduce lyrics etc.
6. Do not name commercial products as competitors; refer to papers and open code only.
7. Budget: about 100 - 150 tool calls; stop searching a sub-topic when returns diminish; spend the depth on the **priority list** of your cluster first, then widen.
8. Mark what the project already tried (see the table in `PROJECT_CONTEXT.md`) in the card field `tried_before`.

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

## Structure of every cluster file (`report_final/text/lit16/<cluster>.md`)
1. **Coverage statement** (5 - 10 lines): what you searched, which sources you could and could not open, which priority papers you read at depth M, which only at A, which you could not find.
2. **Cards** (format above), ordered by relevance to our design; prefix as given in your brief.
3. **Cluster compatibility notes:** (a) interface contracts: what representation flows into and out of this stage in the literature and how it maps to our cached BEV pyramid and our vertex / token / validity / edge output; (b) known adapters and known conflicts; (c) combinations that the papers themselves demonstrate (e.g. "X was used with Y in paper Z").
4. **Reuse versus re-implement:** a short list with reasons, given our environment (no mmcv / mmdet, cached features).
5. **Open questions and gaps:** what you could not determine, what the papers do not say, what no paper does.
6. **Search log:** the queries you ran (one line each) so the search can be reproduced and extended.
Length: no limit on the number of cards, but every card must be useful; a cluster file may be 300 - 700 lines.
