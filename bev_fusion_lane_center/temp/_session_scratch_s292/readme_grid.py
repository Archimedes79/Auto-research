p = "training/grid/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("| `qgraph.py` |"))
assert L[i].endswith(" |") and "E292" not in L[i]
L[i] = L[i][:-2] + "; E292 `QGraph(soft_nms=['mnms', R, sigma])` (the Matrix-NMS layer in the same place, no parameters; phase 9 = nodes / links / map encoder / assigner on a FROZEN pyramid; `_toy_batch` is the synthetic batch of the selftest and of `test_mnms_layer.py`) |"
new = [
    "| `qmnms.py` | E292: the one-pass Matrix-NMS decay as a torch layer on the 2 m node grid (`matrix_nms_logit(e_logit, pos, radius, sigma)`, computed in log space without a clamp, so the gradient is alive at any logit); the twin of `qgraph_readout.matrix_nms` (E288) | new | `python qmnms.py` (against the numpy function at three settings, gradients, extreme logits) |",
    "| `test_mnms_layer.py` | E292: the layer as a model switch (arguments, checkpoint round trip incl. the config.json of a running arm, no new parameter), phase 9 (trainable groups, no gradient into the pyramid, the numbers-only gate), the loss sees the damped e', the mutations of the card (order sign, compensation term, radius, node loss reading e_raw, pyramid not frozen); `--real`: the kill criterion of the card, the layer reproduces `matrix_nms` on 20 frames of B's full dump | new | `python test_mnms_layer.py [--real]` (`--real` needs `temp/e276/predfull_e273_p_s0.pkl`: run_all_tests.sh) |",
]
L[i + 1:i + 1] = new
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("ok")
