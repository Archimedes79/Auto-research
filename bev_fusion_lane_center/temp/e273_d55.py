"""one-off: E273 card revision 5: D55 (matching in the loss is allowed, the rule is end-to-end differentiable); two graph-stage arms"""
p = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(p, encoding="utf-8").read()
rep = [
    ("If this fails its gate, the fallback is bipartite matching in the LOSS only (the literature's standard; the forward path stays differentiable) - that needs Mirko's decision (PROBLEM.md says \"no matching\").",
     "**D55 (Mirko, 2026-11-04): the rule is \"end-to-end differentiable\"; matching inside the loss is allowed.** So the port graph's location-based targets are one way to train it, not a requirement: the graph stage has two arms (section 2.2b), and the better one on the VAL numbers is the base."),
    ("2. Matching-free (ports as key points, pairs by location); bipartite matching in the loss only as the fallback (needs a D entry, PROBLEM.md says \"no matching\"). OK?",
     "2. (answered by D55: matching in the loss is allowed.) Graph stage: arm P (port graph, section 2.2) and arm Q (CGNet-style query decoder on our frozen features, section 2.2b) both trained on the same pyramid head, the better one on VAL becomes the base. OK?"),
    ("- **Targets** (no matching): ports from the GT pieces",
     "- **Targets** (by location, no matching needed; D55 allows matching but this arm does not use it): ports from the GT pieces"),
]
for o, n in rep:
    assert o in s, o[:70]
    s = s.replace(o, n)
b = s.index("### 2.3 ")
new = """### 2.2b Arm Q: a query decoder with matching in the loss (D55, revision 5)
The field's standard answer for lane graphs (TopoNet, MapTRv2, LaneSegNet, LaneGAP, CGNet, ECCV 2024, nuScenes, code `xiaomi/cgnet`, Apache-2.0): a fixed set of queries (about 100-200) reads the BEV features by (deformable) attention; each query outputs one lane piece as a Bezier curve (control points), an existence score and its attributes; a pair head over the queries gives a soft successor matrix (the topology). The forward path is fully differentiable; Hungarian matching only decides, in the loss, which query is compared with which GT piece (one-to-one for the pieces; RATopo, ACM MM 2025, adds one-to-many supervision for the topology). CGNet adds three modules aimed at our weakness: junction-aware queries (a positional prior for split / merge points), a Bezier-space continuity constraint between connected pieces, and iterative topology refinement by a graph network with memory.
- **For us**: the decoder reads the pyramid head's features (0.5 / 1 / 2 m) instead of its own camera encoder, so training is a few GPU hours, not 110 epochs. Targets: the GT pieces after the pass-through join (the GT chat's step 1), each as one Bezier (99.5 % within 0.5 m, measured), successor matrix from the GT edges.
- **Smoothness and connectivity** come from the Bezier representation (a piece is a smooth curve, connected pieces share end pose through the continuity loss), as in CGNet.
- **Differentiable up to the id and the map id**: the queries are the pieces, their ids are their indices, and the assignment of section 2.4 is a cross-attention from the queries to the map pieces.
- **Risk** (named in section 1b): E32 (a small DETR-style decoder from scratch, loops 2-3) lost 0.058 F1 to the grid method; CGNet is the mature version, but its numbers on nuScenes were not read from the paper yet (the abstract claims state of the art without numbers); the first step is to read its tables and, if its checkpoint runs on nuScenes images, to score its graphs with our lane suite as a reference row next to d.
- **Arms**: P (section 2.2, ports + soft connections) and Q (this section) share the pyramid head, the targets of the GT chat, the map encoder and the assignment; they differ only in the graph stage, so the comparison is clean. Verdict by the rule of section 0 on the graph guards (junction F1, topo F1, F1@1.0) first, then on `lane_acc_err`.

"""
s = s[:b] + new + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
