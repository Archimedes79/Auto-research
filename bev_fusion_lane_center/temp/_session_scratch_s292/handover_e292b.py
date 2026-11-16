p = "HANDOVER.md"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("3. **E292**"))
L[i] = ("3. **E292** RESULT 2026-11-14 (card sections 8 / 9, ledger E292.c2 / E292.a): c2 (fine-tune of B without the layer, pyramid frozen) passes the precondition and is itself +0.0065 / +0.0020 snapped topo F1 over E312 (graph-only +0.0087 / +0.0122): the fine-tune schedule alone helps. **Arm a (Matrix-NMS layer, plain read-out) by the written rule TIE, kept and refined (D87), but the price is large: topo F1 -0.0265 / -0.0231 against c2, junction F1 -0.09 / -0.08, vertices +4.3 per 100 m; the head places more nodes less exactly (recall +0.03, position error +0.035 m, junction precision down); NOT adopted as the delivered path, the close-node step stays; COST not measured (low priority; `temp/e292/cost_ab.py` needs an idle GPU).** "
        "Layer code stays (`qmnms.py`, `QGraph(soft_nms=['mnms', R, sigma])`, phase 9, `test_mnms_layer.py`; log-space layer, no dead gradient: gradient norms equal c2's). No follow-up arm started (compute); the cheap next step if wanted: CPU diagnosis on `temp/e292/predfull_e292_{a,c2}_s<S>.pkl` (where the extra nodes sit), then R 1.25 m or protected junction types (0.6 GPU h per run). "
        "Open decision (Mirko): whether c2 (B + 8250 fine-tune steps) replaces B in the system of record: nothing done.")
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L))
print("ok")
