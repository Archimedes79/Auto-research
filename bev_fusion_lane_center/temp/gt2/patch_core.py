p = "training/grid/jgt.py"
s = open(p, encoding="utf-8").read()
s = s.replace('BUF, CLOSE, SIG = 1.8, 2.0, 1.5', 'BUF, CLOSE, SIG = 1.8, 2.0, 1.5\nCORE_BUF = 0.9                                                         # conflict core: connectors of DIFFERENT pairs closer than 2 x CORE_BUF (parallel lanes of one maneuver stay apart)')
old = "def _angle(u, v):"
new = '''def _core(runs, eid, xid):
    """the conflict core: where connector paths of different (entry, exit) pairs cross, merge or diverge; the union of the pairwise overlaps of the connectors grown by CORE_BUF, grown by 1 m. None when there is none"""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    bufs = [LineString(r["pts"]).buffer(CORE_BUF, cap_style=2) for r in runs]
    keys = list(zip(eid, xid))
    parts = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            if keys[i] != keys[j] and bufs[i].intersects(bufs[j]):
                parts.append(bufs[i].intersection(bufs[j]))
    if not parts:
        return None
    u = unary_union(parts).buffer(1.0)
    return u if not u.is_empty else None


def _angle(u, v):'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace("    zones = _zones(runs)\n", "    zones = _zones(runs)\n    core = _core(runs, eid, xid)\n", 1)
s = s.replace("                order=dict(maneuvers=len(man), multi=n_multi, monotone=n_mono, identity=n_ident))", "                core=core, order=dict(maneuvers=len(man), multi=n_multi, monotone=n_mono, identity=n_ident))", 1)
s = s.replace("    zone = np.zeros((n, n), np.uint8)\n    heat", "    zone = np.zeros((2, n, n), np.uint8)\n    heat", 1)
s = s.replace("        cv2.fillPoly(zone, [np.round(pts).astype(np.int32)], 1)\n", "        cv2.fillPoly(zone[0], [np.round(pts).astype(np.int32)], 1)\n", 1)
old2 = "    ii, jj = np.meshgrid("
new2 = '''    if jg.get("core") is not None:
        for q in (jg["core"].geoms if hasattr(jg["core"], "geoms") else [jg["core"]]):
            if q.geom_type == "Polygon" and not q.is_empty:
                cv2.fillPoly(zone[1], [np.round(((np.asarray(q.exterior.coords, float) + half) / res)[:, ::-1]).astype(np.int32)], 1)
    ii, jj = np.meshgrid('''
s = s.replace(old2, new2, 1)
s = s.replace("the 1 m maps of the head, row = x, column = y as `qtargets`: zone (n, n),", "the 1 m maps of the head, row = x, column = y as `qtargets`: zone + conflict core (2, n, n),")
s = s.replace("    fields(jg, n=100, res=1.0)    the 1 m maps of the head: zone, entry heat", "    fields(jg, n=100, res=1.0)    the 1 m maps of the head: zone and conflict core (2 channels), entry heat")
s = s.replace('    jg["order"]                   dict(', '    jg["core"]                    shapely geometry or None: the conflict core, where connectors of different pairs cross, merge or diverge (`_core`)\n    jg["order"]                   dict(', 1)
s = s.replace("    zn, heat, head = fields(jg)\n    assert zn.sum() > 50", "    zn, heat, head = fields(jg)\n    assert zn[0].sum() > 50 and zn.shape[0] == 2")
open(p, "w", encoding="utf-8").write(s)

q = "training/grid/qjunction.py"
t = open(q, encoding="utf-8").read()
t = t.replace('t["jz"]      (1, 100, 100) float32   junction zone (1 m cells, row = x, column = y, origin -50 m)', 't["jz"]      (2, 100, 100) float32   junction zone and conflict core (1 m cells, row = x, column = y, origin -50 m)')
t = t.replace("return dict(jz=zone[None], jheat=heat", "return dict(jz=zone, jheat=heat")
t = t.replace('t["jz"].shape == (1, 100, 100)', 't["jz"].shape == (2, 100, 100)').replace('t["jz"].sum() > 20', 't["jz"][0].sum() > 20')
t = t.replace('t0["jz"].sum() == 0', 't0["jz"].sum() == 0')
open(q, "w", encoding="utf-8").write(t)

m = "training/grid/jmaps.py"
u = open(m, encoding="utf-8").read()
u = u.replace("(B, 7, 100, 100)   1 m logits: zone, entry heat", "(B, 8, 100, 100)   1 m logits: zone, conflict core, entry heat")
u = u.replace("NCH1 = 7", "NCH1 = 8                                                                 # zone, core, entry heat, exit heat, entry heading (2), exit heading (2)")
u = u.replace("[-3.0, -4.0, -4.0, 0.0, 1.0, 0.0, 1.0]", "[-3.0, -3.0, -4.0, -4.0, 0.0, 1.0, 0.0, 1.0]")
u = u.replace("l_z = F.binary_cross_entropy_with_logits(j1[:, :1], jz) + _dice(j1[:, :1], jz)", "l_z = F.binary_cross_entropy_with_logits(j1[:, :2], jz) + _dice(j1[:, :2], jz)")
u = u.replace("F.binary_cross_entropy_with_logits(j1[:, 1:3], jh, pos_weight=pw) + _dice(j1[:, 1:3], jh)", "F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw) + _dice(j1[:, 2:4], jh)")
u = u.replace("pr = j1[:, 3 + 2 * k: 5 + 2 * k]", "pr = j1[:, 4 + 2 * k: 6 + 2 * k]")
u = u.replace('heat = out["j1"][:, 1:3].float().sigmoid()', 'heat = out["j1"][:, 2:4].float().sigmoid()')
u = u.replace('t = out["j1"][b, 3 + 2 * k: 5 + 2 * k, i, j].float()', 't = out["j1"][b, 4 + 2 * k: 6 + 2 * k, i, j].float()')
u = u.replace('jz=torch.zeros(2, 1, 100, 100)', 'jz=torch.zeros(2, 2, 100, 100)').replace('assert out["j1"].shape == (2, 7, 100, 100)', 'assert out["j1"].shape == (2, 8, 100, 100)')
u = u.replace('batch["jz"][:, :, 40:60, 40:60] = 1', 'batch["jz"][:, :, 40:60, 40:60] = 1')
u = u.replace('o = dict(j1=torch.full((1, 7, 100, 100), -6.0))\n    o["j1"][0, 1, 30, 50] = 6.0\n    o["j1"][0, 4, 30, 50] = 5.0\n    o["j1"][0, 3, 30, 50] = 0.0', 'o = dict(j1=torch.full((1, 8, 100, 100), -6.0))\n    o["j1"][0, 2, 30, 50] = 6.0\n    o["j1"][0, 5, 30, 50] = 5.0\n    o["j1"][0, 4, 30, 50] = 0.0')
open(m, "w", encoding="utf-8").write(u)

e = "evaluation/jmaps_eval.py"
v = open(e, encoding="utf-8").read()
v = v.replace("fake = dict(j1=torch.cat([torch.zeros_like(ev[:, :1]), ev[:, 2:3], ev[:, 3:4], torch.zeros_like(ev[:, :1]).repeat(1, 4, 1, 1)], 1))", "fake = dict(j1=torch.cat([torch.zeros_like(ev[:, :2]), ev[:, 2:3], ev[:, 3:4], torch.zeros_like(ev[:, :1]).repeat(1, 4, 1, 1)], 1))")
v = v.replace('''            pz = (out["j1"][:, :1].float().sigmoid() > 0.5).float()
            acc.add("zone", tp=float((pz * jz).sum()), fp=float((pz * (1 - jz)).sum()), fn=float(((1 - pz) * jz).sum()))''', '''            pz_all = (out["j1"][:, :2].float().sigmoid() > 0.5).float()
            for c, name in enumerate(("zone", "core")):
                pz, tz = pz_all[:, c:c + 1], jz[:, c:c + 1]
                acc.add(name, tp=float((pz * tz).sum()), fp=float((pz * (1 - tz)).sum()), fn=float(((1 - pz) * tz).sum()))''')
v = v.replace('''    z = acc.k["zone"]
    res["zone_iou"] = z["tp"] / max(z["tp"] + z["fp"] + z["fn"], 1)
    res["zone_f1"] = 2 * z["tp"] / max(2 * z["tp"] + z["fp"] + z["fn"], 1)''', '''    for name in ("zone", "core"):
        z = acc.k[name]
        res[f"{name}_iou"] = z["tp"] / max(z["tp"] + z["fp"] + z["fn"], 1)
        res[f"{name}_f1"] = 2 * z["tp"] / max(2 * z["tp"] + z["fp"] + z["fn"], 1)''')
v = v.replace("zone IoU {res['zone_iou']:.3f} F1 {res['zone_f1']:.3f};", "zone IoU {res['zone_iou']:.3f} F1 {res['zone_f1']:.3f}; core IoU {res['core_iou']:.3f} F1 {res['core_f1']:.3f};")
open(e, "w", encoding="utf-8").write(v)
print("patched")
