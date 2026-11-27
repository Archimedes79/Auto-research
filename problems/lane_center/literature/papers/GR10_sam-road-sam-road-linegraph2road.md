# GR10 - SAM-Road / SAM-Road++ / LineGraph2Road (vertex mask NMS + pairwise edge classifier with interior samples)

*Pass: loop 16 (2026-10-23), cluster `growth` (iterative growth of road / lane graphs (agents, walkers, trackers)). Reading depth: **M+C (SAM-Road README); M (the other two)** (A abstract, M method, C code page). Verified: checked (three paper HTML pages, README).. Pipeline stage: S6, S3, S2, T.*

## Papers

Segment Anything Model for Road Network Graph Extraction (Hetang et al., CVPRW 2024) https://arxiv.org/abs/2403.16051 ; Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method (SAM-Road++, Yin et al., arXiv 2411.16733; CVF CVPR 2025 listing seen in a search result) https://arxiv.org/abs/2411.16733 ; LineGraph2Road: Structural Graph Reasoning on Line Graphs for Road Network Extraction (2026) https://arxiv.org/abs/2602.23290

## Idea

Not growers but the strongest "link" counter-design on the same task. Vertices come from road / intersection masks by NMS; every vertex looks at its 16 nearest neighbours within 64 px; a small transformer scores each pair from features sampled at both ends. SAM-Road++ adds two things that matter here: (1) *node-guided resampling* - training pairs use the vertices that NMS would choose at inference (distribution match, an exposure-bias fix for link heads); (2) *extended-line samples* - mask values sampled along the extension of both ends (n = 15, 8 px) and along the segment (m = 20) are fed to the classifier (occlusion, ray-style verification). LineGraph2Road builds edge features from the two endpoints + 4 interior bilinear samples and runs a graph transformer on the line graph (edges as nodes).

## Formulation

SAM-Road: pair input = bilinear features at source and target + relative offset -> 3 self-attention layers -> sigmoid; positive = immediate BFS neighbours; losses BCE (masks and edges, equal weights); teacher forcing on GT graphs subdivided and NMS'd to mimic inference. Inference by 16 x 16 sliding windows with averaging. LineGraph2Road: Coupled NMS radii 8 / 16 px, candidate pairs within 64 px, 3-layer graph transformer (4 heads, hidden 128), loss BCE masks + 0.1 BCE edges.

## Input

aerial RGB 512 x 512 (City-scale) or 256 x 256 (SpaceNet), 1 m / px; `image_feat` pyramid (SAM encoder).

## Output

`graph(nodes+edges)`; edge probabilities.

## Training

`teacher_forcing` / `matching_free` BCE; one RTX 4090.

## Differentiable?

partial - edge scoring is differentiable; NMS for vertices and thresholds are not.

## Sensors and data

aerial; CityScale (27 test tiles) and SpaceNet; SAM-Road++ adds a global-scale set.

## Reported results

SAM-Road CityScale 77.23 TOPO-F1 / 68.37 APLS, 4.6 min vs RNGDet++ 231.0 min (Table 2 of SAM-Road). SAM-Road++ ablation (rows match the SpaceNet numbers; dataset not stated in the table; APLS / F1): base 71.64 / 80.52; resampling only 71.90 / 81.77; extended-line only 73.22 / 80.89; both 73.44 / 81.57 (Table 4). LineGraph2Road CityScale 83.77 / 70.40, SpaceNet 84.24 / 73.94 (best variants). SAM-Road ablations: no transformer in the topology decoder 73.75 / 59.39, no intersection mask 71.94 / 64.62, A* instead of learned edges 69.21 / 63.32.

## Compute

edge decoder small; a 4 x 4 window stride cuts time to 2.9 min with F1 77.00.

## Code

https://github.com/htcr/sam_road ; https://github.com/earth-insights/samroadplus ; https://github.com/wzzzzzzy/LineGraph2Road

## Fit to our project

reuse_idea + baseline - effort S: interior / extension samples along a candidate segment as classifier input is the clearest published support for verifying long rays by intermediate samples; node-guided resampling is the cheap exposure-bias fix for any pairwise head.

## Needs

candidate vertex set (NMS), a pairwise head, a distance cap (64 px).

## Conflicts

replaces growth by linking (so it is the alternative family, not a part of the tree); SAM encoder not available (frozen BEV pyramid instead).

## Tried in our experiments

E42 / E45 / E46 / E51 / E56 (learned link heads, TopoLogic-style scoring, path repair) were within 0.01 of the rules and dropped; E113 / E116 (junction / piece verifier on the learned graph) lifted split / merge F1 from 0.385 / 0.428 to 0.437 / 0.504 - a verifier trained on the trace network's own proposals, which is what node-guided resampling would formalise.
