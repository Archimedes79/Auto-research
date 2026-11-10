# sept25 details: method, results, what is not shown, code

Source for everything below: https://arxiv.org/html/2505.12246v1 (read 2026-11-10 through the summarising page fetch tool; section and table names are the page's). Items marked "not stated" were asked for and are not on the page.

## 1. Method at tensor level

| step | input -> output | notes |
|---|---|---|
| BEV encoder (host) | multi-view images -> F_B in R^(H x W x C), H = 200, W = 100 | C not stated; host = TopoNet, TopoLogic, LaneSegNet official code, ResNet-50 |
| SD raster | polylines -> binary occupancy H x W x (road types) -> CNN -> F_SD^R in R^(H x W x C) | CNN not stated; roads, crosswalks, sidewalks |
| SD vector | polylines -> M segments, fixed points per segment (number not stated) -> transformer encoder (as SMERF) -> F_SD^V in R^(M x C) | |
| FT (Eq. 1) | F_B, F_SD^R -> project along channels, difference F_delta in R^(H x W x C), max-pool to R^C, FiLM -> gamma, beta in R^C; F_B^R = gamma * F_SD^R + beta | misalignment reduction, raster side |
| cross-attention | queries F_B (H*W cells), keys / values F_SD^V -> F_B^V in R^(H x W x C) | alignment implicit; deformable attention not mentioned |
| DGFF (Eq. 2-4) | concat(F_B^R, F_B^V) -> FFN -> fused; sigmoid gates g_R, g_V from F_B^R, F_B^V; each gate times fused -> own projection; output = mu * proj_R + nu * proj_V | mu = nu = 0.5 best (Table V); the gates come from the unfused features and act on the fused one, the page does not explain why |
| IKPD head | fused F_B^SD -> residual CNN blocks (depthwise-separable conv, dilated conv, squeeze-and-excitation) -> 1x1 conv, sigmoid -> heat in R^(H x W x 1) | target: Gaussians at SD-map merge, diverge and cross points; radius "reflecting spatial uncertainty", value not stated; targets come from the SD map, not the HD map |
| host heads | F_B^SD -> lane segment / area / topology heads | topology head and L_TOP unchanged and not described; adjacency matrix over the lane graph |

Loss (Eq. 5): L = L_DET + L_TOP + L_IKPD, no weights given; L_IKPD = focal loss (variant not stated). Fusion happens once on the final BEV feature, not per encoder layer. Training: 8 GPUs, total batch 8; optimiser, learning rate, epochs, image size, query count not stated ("as the baselines"). Data: OpenLane-V2 subset_A (Argoverse 2, 1k scenes, six cities, 2 Hz, about 27k train and 4.8k val frames); SD maps "extracted from OpenStreetMap".

## 2. Results (OpenLane-V2 subset_A val, ResNet-50; no seeds, no intervals; deltas are the paper's, against the same host without SD map)

Table I (OLS metric; the columns after DET_t are v1.0 TOP_ll / TOP_lt / OLS, then v1.1 TOP_ll / TOP_lt / OLS):

| method | SD | DET_l | DET_t | v1.0 TOP_ll | TOP_lt | OLS | v1.1 TOP_ll | TOP_lt | OLS | params |
|---|---|---|---|---|---|---|---|---|---|---|
| TopoNet | no | 28.6 | 48.6 | 4.1 | 20.3 | 35.6 | 10.9 | 23.8 | 39.8 | 62.6 M |
| TopoNet + SMERF | yes | 33.4 | 48.6 | 7.5 | 23.4 | 39.4 | 15.4 | 25.4 | 42.9 | 65.8 M |
| TopoNet + SEPT | yes | 34.2 | 49.8 | 8.3 | 23.8 | 40.4 | 19.5 | 27.5 | 45.2 | 70.4 M |
| TopoLogic | no | 29.2 | 46.5 | 18.0 | 20.6 | 40.9 | 23.6 | 24.2 | 43.4 | 61.8 M |
| TopoLogic + SMERF | yes | 31.0 | 48.7 | 21.2 | 22.4 | 43.3 | 26.9 | 26.2 | 45.7 | 65.1 M |
| TopoLogic + SEPT | yes | 34.3 | 48.9 | 25.1 | 25.1 | 45.8 | 31.2 | 29.7 | 48.4 | 69.6 M |

The page also lists TopoNet with the OLV2 raster, OSMG and OSMR SD variants of another paper (OLS 36.1 to 37.7, v1.0 only). Printed deltas of TopoNet + SEPT: TOP_ll +4.2 (v1.0), +8.6 (v1.1). Computed: TOP_ll relative gain v1.1 is +79 % (TopoNet) and +32 % (TopoLogic); SEPT over SMERF on v1.1 is +4.1 and +4.3 TOP_ll.

Table II (LaneSegNet host, OLUS metric; DET_ls / DET_a / DET_te / TOP_lsls / TOP_lste / OLUS / params):

| variant | DET_ls | DET_a | DET_te | TOP_lsls | TOP_lste | OLUS | params |
|---|---|---|---|---|---|---|---|
| LaneSegNet (no SD) | 30.9 | 20.0 | 36.7 | 25.6 | 20.8 | 36.7 | 61.8 M |
| raster only | 33.8 | 28.1 | 38.1 | 27.5 | 21.8 | 39.9 | 65.5 M |
| vector only | 35.3 | 22.3 | 39.2 | 30.2 | 22.6 | 39.9 | 65.8 M |
| hybrid fusion (no IKPD) | 35.8 | 28.2 | 39.6 | 31.0 | 22.7 | 41.4 | 69.8 M |
| SEPT (hybrid + IKPD) | 38.4 | 29.0 | 40.0 | 32.2 | 23.8 | 42.6 | 70.4 M |

Other ablations: Table III, FT module on the raster branch: OLUS 38.2 to 39.9 (TOP_lsls 26.7 to 27.5). Table IV, fusion type (TOP_lsls / OLUS): addition 29.5 / 39.4, concatenation 29.9 / 39.9, cross-attention 30.0 / 40.3, DGFF 31.0 / 41.4. Table V, raster / vector weights 0.2 / 0.8, 0.5 / 0.5, 0.8 / 0.2: OLUS 40.8, 41.4, 40.7. No speed or FPS anywhere. The IKPD effect exists only as the difference between rows "hybrid" and "SEPT".

## 3. What is not shown

- Any split of TOP_ll by junction vs plain road, by distance, or by pair type; any per-pair or per-junction number.
- IKPD without the SD map, IKPD on TopoNet or TopoLogic, IKPD radius or loss variants.
- SD-map noise, shift, rotation, outdated or missing map (one qualitative outdated-map example only, where the model follows the cameras); SD-off at test time.
- Seeds, intervals, speed, memory; whether the baselines of Table I were re-run (values differ a little from the original TopoLogic table, see note.md, Doubts).
- subset_B (nuScenes-based) and any geographically disjoint split.
- The v1.0 TOP metric has "a potential loophole issue" (page); the v1.1 columns are the ones to quote.

## 4. Code

No repository for SEPT: none linked on the arXiv abstract page, the HTML page, the Hugging Face paper page or the Emergent Mind page; a GitHub repository search and the GitHub search API for the title returned 0 results; the guessed profile https://github.com/muleilan returned 404 (not evidence of anything). The paper says it keeps the official implementations of the hosts and adds only the SD modules, so a port starts from:
- https://github.com/NVlabs/SMERF (vector SD encoder and OSM preparation, `openlanev2/sd_maps/load_sdmap_graph.py`; mmdet3d 1.0.0rc6; LICENSE file present, type not named in the README; local digest PL43 calls it non-commercial);
- https://github.com/OpenDriveLab/LaneSegNet (Apache-2.0 per the README; the host of Table II);
- TopoLogic repository (https://github.com/Franpin/TopoLogic): no licence found in the local digest TP01.
What is reusable: nothing from SEPT directly; ideas and equations (FT, DGFF, IKPD) can be written from the page in under 100 lines of PyTorch. The vector side is already in our E273 map encoder.
