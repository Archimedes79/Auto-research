# literature - the papers and code sources of the camera + radar BEV system

| file / folder | content |
|---|---|
| `papers.csv` | one row per paper (id, title, url, axis, tier, verdict, reason, text, idea, extract, code, licence) |
| `papers/` | 42 notes, the ones that code, cards or the SOA digest cite: what the paper does, what is portable. The other 174 of the original 216 were removed on 2026-11-03 at Mirko's word (D49; nothing read them); they are in the git tag `archive/docs-before-cleanup-2026-11-03` (`git checkout archive/docs-before-cleanup-2026-11-03 -- bev_fusion/literature/papers/<file>`). `papers.csv` still lists every paper of the scans, 68 of its rows never had a note |
| `code/front_bev_code_sources.md` | repository, licence and module per block (read on the repository pages, nothing downloaded) |
| `fixes_for_gate_findings.md` | fixes for the four findings of the synthetic gates (scout of 2026-10-25) |
| `radar_stationary_gt.md` | radar stationary-structure target: how others supervise it, which GT can be reused, decision table (scout of 2026-10-26) |

The raw tables of the scouts are in `scout_tables/`. Papers of the lane project are in `../../bev_fusion_lane_center/literature/`.
