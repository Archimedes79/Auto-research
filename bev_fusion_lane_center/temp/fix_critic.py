"""one-off: the fixes of the critic pass on the E273 lane head code (findings 1-5 of the report of 2026-11-04 night)"""
import re


def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:70])
    s = s.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(s)


# 2. structure term: a long link crosses a suppressed connector piece, so the map pieces are joined by TWO successor steps
sub("training/grid/qgraph.py", "        C[:, :K, :K] += batch[\"map_adj\"].to(dev)[pb, 0].float()\n",
    "        A1 = batch[\"map_adj\"].to(dev)[pb, 0].float()\n        C[:, :K, :K] += A1 + A1 @ A1                                                           # a long link crosses a suppressed connector piece: two successor steps\n        C.clamp_(max=1.0)\n")
# 3. registration: atan2(0, 0) has a NaN backward
sub("training/grid/qassign.py", "th = torch.atan2(sxy - syx, sxx + torch.einsum(\"bat,bay,bty->b\", W, pc, tc)).clamp(", "th = torch.atan2(sxy - syx, sxx + torch.einsum(\"bat,bay,bty->b\", W, pc, tc) + 1e-6).clamp(")
# 5. sampling in float32 (bf16 grid: 0.2 m steps beyond 25 m)
sub("training/grid/qnodes.py", "    grid = torch.stack([xy[:, 1] / HALF, xy[:, 0] / HALF], -1).to(feat.dtype)\n    return F.grid_sample(feat, grid, mode=\"bilinear\", padding_mode=\"border\", align_corners=False)",
    "    grid = torch.stack([xy[:, 1] / HALF, xy[:, 0] / HALF], -1).float()                         # float32 even under bf16 autocast (a bf16 grid has 0.2 m steps beyond 25 m)\n    return F.grid_sample(feat.float(), grid, mode=\"bilinear\", padding_mode=\"border\", align_corners=False).to(feat.dtype)")
sub("training/grid/qlinks.py", "                ev[sel] = sample(cen[b: b + 1], xy.to(cen.dtype))[0, 0]", "                ev[sel] = sample(cen[b: b + 1], xy)[0, 0].to(ev.dtype)")
# focal alpha 0.5: no class-prior shift of the 0.5 read-out threshold
sub("training/grid/qgraph.py", "def focal_bce(logit, y, alpha=0.25, gamma=2.0, norm=None):", "def focal_bce(logit, y, alpha=0.5, gamma=2.0, norm=None):")
# phases: node loss only where the node head is trained, link loss only where the links are trained
sub("training/grid/qgraph.py", "    if phase in (3, 4, 5, 6, 7):\n        l, i = node_loss(out[\"no\"], batch)", "    if phase in (3, 5, 7):\n        l, i = node_loss(out[\"no\"], batch)")
sub("training/grid/qgraph.py", "    if phase in (4, 6, 7):\n        l, i = link_loss(out[\"L\"], batch)", "    if phase in (4, 7):\n        l, i = link_loss(out[\"L\"], batch)")
# the intersection only is dilated (the road polygons are not)
sub("training/grid/qraster.py", "def areas2(g, n=50, res=2.0, dilate=0.0, ss=4):", "def areas2(g, n=50, res=2.0, dilate=0.0, ss=4, dilate_road=0.0):")
sub("training/grid/qraster.py", "    road, inter = _polygons(a.get(\"road\", []), dilate), _polygons(a.get(\"intersection\", []), dilate)", "    road, inter = _polygons(a.get(\"road\", []), dilate_road), _polygons(a.get(\"intersection\", []), dilate)")
sub("training/grid/qraster.py", "    \"\"\"-> (2, n, n) float32: channel 0 road (cell covered >= 0.5 by the drivable polygons, ss x ss sub-cells), channel 1 intersection (any part of the cell covered)\"\"\"",
    "    \"\"\"-> (2, n, n) float32: channel 0 road (cell covered >= 0.5 by the drivable polygons, ss x ss sub-cells), channel 1 intersection (any part of the cell covered); `dilate` grows the INTERSECTION polygons only\n    (the loader's 2 m), `dilate_road` the road polygons (default 0)\"\"\"")

# 1 + 4. gates: fail closed, a NaN is not a pass, phase 4 reads the graph without the (not yet trained) section head, phase 6 tolerances, a zero-initialised geometric baseline
p = "training/train/qgates.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    try:
        return fn(model, run, n)
    except Exception as e:
        import traceback
        run.log("gate error", traceback.format_exc()[-800:])
        return {"pass": True, "why": f"gate could not be read: {e!r}"}''', '''    try:
        return fn(model, run, n)
    except Exception as e:                                                                   # fail CLOSED: a gate that cannot be read stops the night (the kill criterion is not enforced otherwise)
        import traceback
        run.log("gate error", traceback.format_exc()[-800:])
        return {"pass": False, "why": f"gate could not be read: {e!r}"}''')
s = s.replace('''        res.update({"pass": True, "why": f"graph gate not read: {e!r}"})''', '''        res.update({"pass": False, "why": f"graph gate could not be read: {e!r}"})''')
s = s.replace("    ok_all = acc >= 0.95 and (ppg is None or ppg != ppg or ppg <= 2.0)", "    ok_all = acc >= 0.95 and ppg is not None and ppg == ppg and ppg <= 2.0")
s = s.replace('''    asg0 = QA.Assigner(d=model.asg.d).to(dev) if geometric else None''', '''    asg0 = QA.Assigner(d=model.asg.d).to(dev) if geometric else None
    if asg0 is not None:
        for m_ in (asg0.wq, asg0.wk, asg0.dust):                                                 # the geometric start assigner has no learned part (E272's start values)
            torch.nn.init.zeros_(m_.weight)
            torch.nn.init.zeros_(m_.bias)''')
s = s.replace('''    ok = res["lane_acc_error"] >= res["geo_acc_error"] and res["lane_acc_exact"] >= res["geo_acc_exact"] - 0.01
    return dict(res, **{"pass": bool(ok), "why": "lane_acc(error) >= geometric start; lane_acc(exact) >= geometric start - 0.01"})''', '''    ok = res["lane_acc_error"] >= res["geo_acc_error"] and res["lane_acc_exact"] >= res["geo_acc_exact"] - 0.03
    return dict(res, **{"pass": bool(ok), "why": "lane_acc(error) >= geometric start; lane_acc(exact) >= geometric start - 0.03 (the card's guard of 0.01 is read on the full scoring)"})''')
# phase 4: the delivered graph is read with the connected link components as sections (the section head is trained in phase 5)
s = s.replace('''def graph_numbers(model, n):''', '''def component_sections(p, thr_e=0.5, thr_link=0.5):
    """sections = the connected components of the kept links (used where the section head is not trained yet)"""
    nd, lk = p["nodes"], p["links"]
    M = len(nd["e"])
    par = list(range(M))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    keep = (np.asarray(nd["e"]) >= thr_e)
    for a, b, pr in zip(lk["src"], lk["dst"], lk["p"]):
        if pr >= thr_link and keep[a] and keep[b]:
            par[f(int(a))] = f(int(b))
    nd["section"] = np.array([f(i) for i in range(M)], int)
    return p


def graph_numbers(model, n, own_sections=True):''')
s = s.replace('''    for b in _batches("exact", n):
        preds += QP.predict(model, b)
    model.train()
    out = QE.score_frames(preds, fr)''', '''    for b in _batches("exact", n):
        preds += QP.predict(model, b)
    model.train()
    assert len(preds) == len(fr), (len(preds), len(fr))
    if not own_sections:
        preds = [component_sections(p) for p in preds]
    out = QE.score_frames(preds, fr)''')
s = s.replace("        gn = graph_numbers(model, n)\n        res.update(gn)\n        ok = (gn[\"junction_F1\"]", "        gn = graph_numbers(model, n, own_sections=False)                                   # the section head is trained in phase 5: sections = link components here\n        res.update(gn)\n        ok = (gn[\"junction_F1\"]")
open(p, "w", encoding="utf-8").write(s)
print("ok")
