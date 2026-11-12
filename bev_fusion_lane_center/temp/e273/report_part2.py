"""one-off: the system-level sections of the Spurmitte report (executed inside build_report.py: it uses the data and charts defined there)"""
import json as _json

# ------------------------------------------------------------------------------------------------------------------ extra data
PIX = _json.load(open(os.path.join(ROOT, "temp", "e274", "pixel_eval.json")))
_sw = os.path.join(ROOT, "temp", "e274", "thr_sweep_e273_p_s0.json")
SWEEP = _json.load(open(_sw)) if os.path.exists(_sw) else None
DIAG = _json.load(open(os.path.join(ROOT, "temp", "e274", "diag_nodes_e273_p_s0.json")))
DIAG2 = _json.load(open(os.path.join(ROOT, "temp", "e274", "diag_nodes2_e273_p_s0.json")))
pm = lambda layer, k: (PIX["e273_p_s0"][layer][k] + PIX["e273_p_s1"][layer][k]) / 2  # noqa: E731


def crop(src_glob, box, name):
    src = sorted(glob.glob(src_glob))[0]
    im = Image.open(src).convert("RGB").crop(box)
    im.save(os.path.join(IMG, name), quality=88, optimize=True)
    return "img/" + name


pix_a = crop(os.path.join(ROOT, "results", "E273.p_s0", "figures", "scene-0221_e273_p_s0_f*.png"), (640, 40, 1500, 480), "pix_0221.jpg")
pix_b = crop(os.path.join(ROOT, "results", "E273.p_s0", "frames", "23", "scene-0092_e273_p_s0_f2[0-9]_*.png"), (640, 40, 1500, 480), "pix_0092.jpg")
pix_c = crop(os.path.join(ROOT, "results", "E273.p_s0", "frames", "27", "scene-0096_e273_p_s0_f2[0-9]_*.png"), (640, 40, 1500, 480), "pix_0096.jpg")

# ------------------------------------------------------------------------------------------------------------------ videos: comparison d vs E273, earlier work
cmp_vid = {}
for idx, name, desc in SC:
    src = os.path.join(OUT, "compare", f"compare_{name}.mp4")
    if os.path.exists(src):
        cmp_vid[name] = webm(src, f"cmp_{name}.webm", width=1120, crf=40)
earlier = []
for rel, name, cap in (("runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val.mp4", "earlier_prototype.webm", "The best graph so far (the prototype recall_sp0_dp: camera + radar features, dense heads, differentiable graph builder, rule assembly): prediction red over ground truth green, two VAL scenes, 3 frames per second. Made in loop 22."),
                       ("runs/grid_view/e271_a_s0/scene7.mp4", "earlier_e271_scene7.webm", "E271, the 1 m lane-centre head with differentiable sparsification: heat map, vertices against ground truth and the prototype, SD-map assignment with the start-value assigner."),
                       ("runs/grid_view/e272_j_s0/scene7.mp4", "earlier_e272_scene7.webm", "E272, the grid chain as one trained function (head, sparsify, hold, assignment): SD map and system side by side in road colours."),
                       ("runs/grid_view/e272_j_s0/scene23.mp4", "earlier_e272_scene23.webm", "E272 on a straight Boston road (scene 23): sparse vertices on the lane, four to six per 100 m on straights.")):
    sp = os.path.join(ROOT, rel)
    if os.path.exists(sp):
        earlier.append((webm(sp, name, width=1120, crf=36), cap))

# ------------------------------------------------------------------------------------------------------------------ charts
pl = ["lane centres (1 m)", "lane dividers (0.5 m)", "road dividers (0.5 m)", "boundaries (0.5 m)", "road (2 m)", "intersection (2 m, grown by 2 m)"]
pk = {"lane centres (1 m)": "lane centres (1 m)", "lane dividers (0.5 m)": "lane dividers (0.5 m)", "road dividers (0.5 m)": "road dividers (0.5 m)", "boundaries (0.5 m)": "boundarys (0.5 m)", "road (2 m)": "road (2 m)",
      "intersection (2 m, grown by 2 m)": "intersection (2 m, grown by 2 m)"}
ch_pix = C.bars(["lane centres", "lane dividers", "road dividers", "borders", "road", "intersection"], ["precision", "recall", "F1"],
                [[pm(pk[k], "P") for k in pl], [pm(pk[k], "R") for k in pl], [pm(pk[k], "F1") for k in pl]], ymax=1.0, fmt=".2f", cls=("s0", "s2", "s1"), width=700, height=290, label_every=False)
evn = ["split", "merge", "stop", "exit", "start", "end"]
ch_pixev = C.bars(evn, ["precision", "recall", "F1"], [[pm(f"event: {e} (1 m)", "P") for e in evn], [pm(f"event: {e} (1 m)", "R") for e in evn], [pm(f"event: {e} (1 m)", "F1") for e in evn]], ymax=1.0, fmt=".2f", cls=("s0", "s2", "s1"), width=560, height=280, label_every=False)

ch_recall = C.bars(["outside intersections", "inside intersections", "plain cells", "event cells", "below 20 m", "20-35 m", "beyond 35 m"],
                   ["recall of true nodes within 2 m"], [[DIAG2[k]["recall_2m"] for k in ("outside", "inside intersection", "plain", "event", "near <20", "mid 20-35", "far >35")]], ymax=1.0, cls=("s0",), width=700, height=280, legend=False, fmt=".2f")
# threshold sweep chart
if SWEEP:
    ks = ["e0.2_l0.5", "e0.3_l0.5", "e0.4_l0.5", "e0.5_l0.5"]
    ths = [0.2, 0.3, 0.4, 0.5]
    ch_sweep = C.lines(ths, ["node recall", "node precision", "topo F1"], [[SWEEP[k]["node_R"] for k in ks], [SWEEP[k]["node_P"] for k in ks], [SWEEP[k]["topo_F1"] for k in ks]], ymax=1.0, ymin=0.0,
                       ref=D["topo_F1"], ref_label=f"reference graph topo F1 {D['topo_F1']:.3f}", width=620, height=300, xfmt="e >= {:.1f}", cls=("s0", "s2", "s1"))
    rows_sw = "".join(f'<tr><td>e &ge; {k.split("_")[0][1:]}, p &ge; {k.split("_")[1][1:]}</td><td class="num">{v["node_R"]:.3f}</td><td class="num">{v["node_P"]:.3f}</td><td class="num">{v["topo_F1"]:.3f}</td><td class="num">{v["F1_1"]:.3f}</td><td class="num">{v["junction_F1"]:.3f}</td>'
                      f'<td class="num">{v["lanes_found"]:.3f}</td><td class="num">{v["dup"]:.3f}</td><td class="num">{v["vert100_dp"]:.1f}</td></tr>' for k, v in SWEEP.items())
    SW_HTML = (f'<div class="grid2"><div class="panel"><h3>Node threshold, no retraining</h3>{ch_sweep}<p class="cap">Seed 0, exact map, 873 VAL frames. Lowering the node threshold raises recall ({SWEEP["e0.5_l0.5"]["node_R"]:.2f} to {SWEEP["e0.3_l0.5"]["node_R"]:.2f} at 0.3), F1@1.0 ({SWEEP["e0.5_l0.5"]["F1_1"]:.3f} to {SWEEP["e0.3_l0.5"]["F1_1"]:.3f}, past the guard 0.848) and lanes found ({SWEEP["e0.5_l0.5"]["lanes_found"]:.2f} to {SWEEP["e0.3_l0.5"]["lanes_found"]:.2f}), but the topology falls ({SWEEP["e0.5_l0.5"]["topo_F1"]:.3f} to {SWEEP["e0.3_l0.5"]["topo_F1"]:.3f}) and duplicates double ({SWEEP["e0.5_l0.5"]["dup"]:.2f} to {SWEEP["e0.3_l0.5"]["dup"]:.2f}): the extra nodes of this head bring false and duplicate links. Recall has to come with precision, which a threshold cannot give.</p></div>'
               f'<div class="panel"><h3>Read-out thresholds (node e, link p)</h3><div class="scroll"><table><thead><tr><th>setting</th><th class="num">node R</th><th class="num">node P</th><th class="num">topo F1</th><th class="num">F1@1.0</th><th class="num">junction F1</th><th class="num">lanes found</th><th class="num">dup</th><th class="num">vert / 100 m</th></tr></thead><tbody>{rows_sw}</tbody></table></div></div></div>')
else:
    SW_HTML = '<p class="cap">The read-out threshold sweep was still running when this page was built.</p>'

# ------------------------------------------------------------------------------------------------------------------ text blocks
def st(kind, text):
    return f'<span class="pill {kind}">{text}</span>'


def row(*cells):
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


LEAD = ("This page puts the whole Spurmitte system in one place: the strategy, the architecture with the literature behind it, the ground-truth work, the results per stage, videos, the weaknesses and the plan. "
        "The newest experiment, E273, is a fully differentiable lane head. By the card's rule it is CUT: the map assignment works, the lane graph under it is not yet good enough. "
        "The work continues with E274 (node recall and long-link matching). By your word of 2026-11-05 (D57) Spurmitte work has priority over everything else, BEV Fusion included.")

STATUS = f"""
<p>Spurmitte is the lane-centre-line graph of the road ahead, built from camera and radar, fused over time, attached to an SD map, and handed to a planner. The system has two tracks. The <b>prototype chain</b> (loops 1-22, dense heads + a differentiable graph builder + rule-based assembly) delivers the best numbers today but is not end-to-end differentiable. The <b>grid chain</b> (E271, E272, E273) is the end-to-end replacement. The gap that decides everything is connectivity at junctions.</p>
<div class="scroll"><table><thead><tr><th>Stage</th><th>Best result so far (VAL 873 frames)</th><th>Where it stands</th></tr></thead><tbody>
{row('<b>Foundation features</b>: frozen GaussianCaR (camera + radar BEV)', 'map-pretrained; lane F1@1.0 rose from 0.624 (CRN) to 0.867 with the earlier heads; cached, so every head trains on it quickly', st('ok', 'fixed'))}
{row('<b>Dense pixel maps</b> (E273 pyramid)', f"F1 per layer: lane centres {pm('lane centres (1 m)', 'F1'):.2f}, lane dividers {pm('lane dividers (0.5 m)', 'F1'):.2f}, road {pm('road (2 m)', 'F1'):.2f}, intersection {pm('intersection (2 m, grown by 2 m)', 'F1'):.2f}; event heat maps only {min(pm(f'event: {e} (1 m)', 'F1') for e in evn):.2f}-{max(pm(f'event: {e} (1 m)', 'F1') for e in evn):.2f} (they over-fire: high recall, low precision)", st('ok', 'good, events weak'))}
{row('<b>Nodes</b> on a 2 m grid (E273)', f"node F1 {m2(lambda s: ND[s]['node_F1']):.2f} at 2 m, median position error {m2(lambda s: ND[s]['pos_err_med']):.2f} m, but recall {m2(lambda s: ND[s]['node_R']):.2f}; E271 vertex head: coverage 0.72-0.73 at 10.3 vertices per 100 m", st('info', 'gate passed, recall short'))}
{row('<b>Links</b> (Hermite curves, E273)', f"short links P {m2(lambda s: ND[s]['link_short_P']):.2f} / R {m2(lambda s: ND[s]['link_short_R']):.2f}; long links through junctions P {m2(lambda s: ND[s]['link_long_P']):.2f} / R {m2(lambda s: ND[s]['link_long_R']):.2f}; junction F1 {mj:.3f} against {D['junction_F1']:.3f}", st('no', 'fails the gate'))}
{row('<b>Sections</b> (one id per lane piece)', f"{m2(lambda s: GR[s]['pieces_per_gt_piece']):.2f} delivered pieces per true piece; only 47-48 % of the section starts right", st('info', 'partial'))}
{row('<b>Delivered graph</b> (E273 vs the best graph so far)', f"topo F1 {mt:.3f} against {D['topo_F1']:.3f}; F1@1.0 {mf:.3f} against {D['F1_1']:.3f}; 2-3 times the vertex budget", st('no', 'CUT'))}
{row('<b>Assignment to the SD map</b>', f"E273: lane accuracy {m2(lambda s: P[s]['assign_error']['lane_acc']):.3f} against {m2(lambda s: G[s]['assign_error']['lane_acc']):.3f} for geometry when the map is 1-2 m off ({sg(mean_err)}); E272: road accuracy 0.984 with an exact map", st('ok', 'works'))}
{row('<b>Temporal hold</b> (E272)', 'built, differentiable, trained; it closes itself on clean frames, so its value is untested (needs degraded frames)', st('info', 'untested'))}
{row('<b>Route attribute and planner consumer</b>', 'old chain only: route head along the edges, route success 0.889; planner consumer path error 0.34 m at 20 m; not part of the new head', st('info', 'not in the new head'))}
{row('<b>Best delivered graph today</b>: recall_sp0_dp (old chain)', f"Q {D['Q']:.3f}, topo F1 {D['topo_F1']:.3f}, junction F1 {D['junction_F1']:.3f}, lanes found {D['lanes_found']:.3f}, F1@1.0 {D['F1_1']:.3f}; rule-based assembly, not end-to-end", st('info', 'reference'))}
{row('<b>Speed</b>', f"E273 head alone {tm('total'):.0f} ms per frame (idle RTX 3060, cached features); old fast chain 20-22 ms; slow control 628 ms", st('ok', 'fast enough'))}
</tbody></table></div>

<h3>The original plan, recapitulated</h3>
<div class="grid2">
<div class="panel"><h3>Your brief of 2026-11-04</h3><ul class="tight">
<li>A pyramidal head: lane markings and road borders at 0.5 m, lane centres at 1 m, roads at 2 m.</li>
<li>Queries, here one node per 2 m cell, with a type: normal, split, merge, entry, exit; smooth shapes by Hermite curves.</li>
<li>A connection matrix between the nodes, and from it sections: one id per lane, new ids only at splits and merges, without a non-differentiable post-processing step.</li>
<li>Assignment of the map ids to the perceived ids, soft, through cross-attention.</li>
<li>Everything end-to-end differentiable up to and including the classification (D55); matching in the loss is allowed.</li>
<li>Local evaluation with pictures, videos and numbers, and one result folder per experiment with head code, GT version, training version, code version, dataset, evaluation code version, qualitative and quantitative results.</li>
</ul></div>
<div class="panel"><h3>E273, the eight steps of the card</h3><div class="scroll"><table><thead><tr><th>#</th><th>Step</th><th>Status</th></tr></thead><tbody>
{row('0', 'targets from the lane GT, map simulator, oracle', st('ok', 'done'))}
{row('1', 'scorer and pictures first (reproduces the best graph exactly)', st('ok', 'done'))}
{row('2', 'pixel maps', st('ok', 'gate passed'))}
{row('3', 'nodes', st('ok', 'gate passed'))}
{row('4', 'links and Hermite curves', st('no', 'gate failed five times'))}
{row('5', 'sections', st('info', 'pieces pass, starts fail'))}
{row('6', 'map encoder and assignment', st('ok', 'gate passed'))}
{row('7', 'joint training, both seeds, scoring, pictures, videos, result folder', st('ok', 'done: CUT'))}
</tbody></table></div></div></div>
<div class="panel" style="margin-top:16px"><h3>The continuation (E274) and what comes after</h3><div class="scroll"><table><thead><tr><th>#</th><th>Step</th><th>Status</th></tr></thead><tbody>
{row('8', 'diagnose where nodes and links are lost (cell-level node misses, read-out thresholds, pixel layers, oracle with nodes removed)', st('ok', 'done 2026-11-05'))}
{row('9', 'write the E274 card: question, prediction, written rule, arms, kill criterion', st('ok', 'done 2026-11-05'))}
{row('10', 'arm r, node recall: existence positives x3 at event cells and x2 inside intersections; code, tests and critic pass done, the launcher starts the training in the 08:30 window', st('info', 'ready'))}
{row('11', 'arm p, long-link matching: a dual-softmax loss over the long-link candidates (entries and exits compete, with a none entry), on the nodes of arm r', st('info', 'ready'))}
{row('12', 'differentiable sparsification on the node chain to meet 14.3 vertices per 100 m', st('info', 'planned'))}
{row('13', 'two seeds, scoring, pictures, videos, report', st('info', 'planned'))}
{row('later', 'temporal fusion by cross-attention to earlier nodes; route attribute; query-decoder reference (CGNet); a TEST check once per finished design, with your recorded OK', st('info', 'after E274'))}
</tbody></table></div><p class="cap">GPU jobs run only inside your windows (nights 22:00-07:00, Monday-Friday 08:30-17:30). D57: the lane work goes first, the BEV system and Image Encoder chats have been told and agreed.</p></div>"""

STRATEGY = """
<ul class="tight">
<li><b>Goal.</b> A planner-ready lane graph without an HD map: lane centre lines with forks and merges, travel direction, fused over time, assigned to the SD-map route (lane level, road level as fallback), with confidence. Very good and very fast: quality at no more than 14.3 vertices per 100 m after Douglas-Peucker 0.3 m, and milliseconds per frame on one RTX 3060.</li>
<li><b>Frozen foundation, light heads.</b> The map-pretrained GaussianCaR features are computed once and cached. Every head trains on them in minutes to hours, so many designs can be compared on the same frames. Nothing in the project trains a camera encoder.</li>
<li><b>Dense evidence first, sparse graph second.</b> A dense pixel stage says where lanes, borders and events are; a sparse stage turns it into nodes and links. A decoder with learned queries trained from scratch lost 0.058 F1 to the dense method earlier (E32), which is why nodes start from dense maps and fixed positions.</li>
<li><b>End-to-end differentiable (D55).</b> The forward path up to and including the map assignment is made of soft steps, so a loss on the output reaches every module. Matching inside the loss (Hungarian, one-to-many, Sinkhorn) is allowed; a hard selection is allowed only on the output and is named every time. Rule steps that remain on the delivered path are listed as gaps in every report.</li>
<li><b>Assignment as a soft, structure-aware step.</b> Published association is hard and at road level. We assign perceived lane nodes to lane-level map pieces with probabilities, a dustbin for nodes without a map piece, and a registration that corrects a shifted map.</li>
<li><b>Measurement discipline.</b> VAL only for choices; TEST once per finished design with your recorded OK; two seeds for a trained arm; paired scene bootstrap; a written rule before every run; every number from a result file; controls named in the card (null, oracle, matched reference).</li>
<li><b>Priority (D57, 2026-11-05).</b> Spurmitte work has priority; all other work, BEV Fusion included, comes second. Applied to the GPU windows and the shared queue; running jobs of other chats are never touched.</li>
</ul>
<h3>The ladder of experiments</h3>
<div class="scroll"><table><thead><tr><th>Step</th><th>What</th><th>Result</th></tr></thead><tbody>
<tr><td>Loops 1-22, prototype chain</td><td>dense heads, differentiable graph builder (D7: seeds, rays, survival, children), rule assembly (path union, fork pruning), hold, route head, planner consumer</td><td>best delivered graph (recall_sp0_dp): Q 0.742, topo F1 0.740; not end-to-end</td></tr>
<tr><td>E271</td><td>1 m lane-centre head on the frozen features + differentiable sparsification</td><td>KEEP by hand: coverage 0.721 / 0.728 and precision 0.871 / 0.873 at 10.3 vertices per 100 m</td></tr>
<tr><td>E272</td><td>the chain as one trained function: head, sparsify, hold, assignment</td><td>TIE by hand: matches E271, hold untested, assignment learned nothing beyond geometry because the map was exact</td></tr>
<tr><td>E273</td><td>pyramid pixel maps, grid nodes, Hermite links, soft sections, map assignment under map errors</td><td>CUT by hand: assignment +0.071, graph guards broken</td></tr>
<tr><td>E274 (planned)</td><td>node recall, path-level link loss, sparsification on the node chain</td><td>card next</td></tr>
</tbody></table></div>"""

# architecture diagram with literature
def lbox(x, y, w, h, title, l1, l2, l3, cls_="box"):
    return (f'<rect class="{cls_}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/><text class="bt" x="{x + 10}" y="{y + 22}">{title}</text>'
            f'<text class="bs" x="{x + 10}" y="{y + 42}">{l1}</text><text class="bs" x="{x + 10}" y="{y + 58}">{l2}</text>'
            f'<text class="bs" x="{x + 10}" y="{y + 82}" style="fill:var(--accent)">{l3}</text>')


def elbow(x0, y0, x1, y1, ym):
    pts = f"{x0},{y0} {x0},{ym} {x1},{ym} {x1},{y1 - 9}"
    return f'<polyline class="arr" points="{pts}"/><polygon class="arrh" points="{x1},{y1} {x1 - 4.5},{y1 - 9} {x1 + 4.5},{y1 - 9}"/>'


boxes = [("Features", "camera + radar BEV, frozen,", "cached (39 channels)", "GaussianCaR, CRN, BEVFormer", "box"),
         ("Pixel pyramid", "0.5 m lines, 1 m centres + events,", "2 m road and intersection", "MGMap, HDMapNet, CenterNet maps", "box hot"),
         ("Grid nodes", "one node per 2 m cell, 8 types,", "offset and heading", "CenterNet, SAM-Road, MaGRoad keypoints", "box hot"),
         ("Hermite links", "short: chord / 3; long: learned", "handles; set attention", "TopoNet, LaneSegNet, CGNet, RATopo", "box hot"),
         ("Sections", "one id per lane piece, new ids", "at splits and merges (soft walk)", "own design", "box hot"),
         ("Temporal hold", "ego-warped memory with a gate", "(E272; closes on clean frames)", "StreamMapNet, MapTracker, TopoStreamer", "box"),
         ("Assignment to the SD map", "map encoder, cross-attention, dustbin,", "soft registration", "SuperGlue, MAT / OMA, SMERF, SEPT", "box hot"),
         ("Route + planner", "route flag along the edges, planner", "consumer (old chain only)", "PlanTF, PLUTO, PDM, OrienterNet", "box")]
sv = ['<svg class="chart" viewBox="0 0 1140 330" role="img" xmlns="http://www.w3.org/2000/svg">']
for i, (t, a, b, c, k) in enumerate(boxes):
    r, col = divmod(i, 4)
    x, y = 4 + col * 292, 10 + r * 180
    sv.append(lbox(x, y, 262, 106, t, a, b, "lit: " + c, k))
    if col:
        sv.append(arrow(x - 30, y + 53, x, y + 53))
sv.append(elbow(4 + 3 * 292 + 131, 116, 4 + 131, 190, 152))
sv.append('<text class="bs" x="4" y="318">blue boxes are trained end to end in E273; grey boxes are built (hold, route, planner) but not part of the E273 head. Blue text: the literature each block builds on.</text></svg>')
ch_arch = "".join(sv)
LIT = [("Features", "GaussianCaR (ours; camera + radar Gaussian lift), CRN, RCBEVDet, BEVFormer", "frozen and cached: 0.624 to 0.867 lane F1@1.0 after the switch to the map-pretrained GaussianCaR", "no paper trains lane heads on frozen cached BEV features"),
       ("Pixel pyramid", "HDMapNet (dense heads), MGMap (multi-scale BEV features), PseudoMapTrainer (finer grids help lines, not areas), CenterNet / CondLaneNet (key points as heat maps)", "three resolutions: lines 0.5 m, centres 1 m, areas 2 m", "event heat maps for splits, merges, stops, exits, starts, ends"),
       ("Grid nodes", "CenterNet (targets by location, no matching), SAM-Road, MaGRoad (key points from a dense map), PolyWorld", "one node per 2 m cell; the target is the point of the true line nearest the cell centre", "nodes inside intersections only at event markers"),
       ("Hermite links", "TopoNet, LaneSegNet, LaneGAP, CGNet (successor matrix, Bezier continuity, junction-aware queries), RATopo (one-to-many topology labels), SAM-Road (link score from path evidence)", "set attention over all active nodes, one-to-many labels (within 1.5 m), Hermite curves with learned handles", "published methods need matching in the loss; ours uses targets by location"),
       ("Sections", "none directly; path-aware attention in MAT / OMA", "a soft absorbing walk along the links: one id per lane piece, a linear solve over 512 nodes", "gap in the literature"),
       ("Temporal hold", "StreamMapNet, MapTracker, TopoStreamer, HRMapNet, MambaMap", "ego-warped memory with a learned gate (E272)", "follow-up: cross-attention from the current nodes to earlier nodes"),
       ("Assignment", "SuperGlue / LightGlue (dustbin), PolyWorld, MAT / OMA (map as polyline tokens), SMERF, SEPT, TopoSD (map noise and priors), OrienterNet", "map encoder over five relations, cross-attention, geometric start value, soft registration, dustbin", "gap: nobody assigns perceived lane-level ids to a lane-level map softly and trains it through the perception"),
       ("Route + planner", "PlanTF, PLUTO, PDM (rule-based centre-line planner), VAD, SparseDrive", "route head along the edges (success 0.889), planner consumer E117 (path error 0.34 m at 20 m)", "no paper back-propagates planning into topology")]
lit_rows = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in LIT)
ARCH = f"""<p>The product chain runs from camera and radar to the planner. Each block below names the literature it builds on. The literature is catalogued in <code>literature/</code> (papers.csv, OVERVIEW, one note per relevant paper); the table repeats the part that shaped each block.</p>
<div class="scroll"><div style="min-width:900px">{ch_arch}</div></div>
<div class="scroll"><table><thead><tr><th>Block</th><th>Literature it builds on</th><th>What we do</th><th>What is different or open</th></tr></thead><tbody>{lit_rows}</tbody></table></div>
<h3>The lane head (E273) in detail</h3>"""

GT = f"""
<p>The ground truth is made by <b>autogt</b>, a standalone suite (it imports nothing from training) that turns raw nuScenes into training-ready bird's-eye-view truth per keyframe. Two stores belong to one release.</p>
<div class="scroll"><table><thead><tr><th>Part</th><th>Content</th><th>Version and state</th></tr></thead><tbody>
<tr><td><b>Front GT</b></td><td>static map layers, object footprints, masks, velocity and flow in m/s (radar-corrected)</td><td>signature <code>c643a7539d</code> (v8, v9 shares it); the cached GaussianCaR features were made with it; the lane head does not read it</td></tr>
<tr><td><b>Lane GT</b></td><td>lane pieces with ids that change only at splits and merges; successors, predecessors, left, right and opposite neighbours; lane index and count; events (split, merge, cross, stop, exit, start, end, window) with in and out ids; segments per piece (inside an intersection or not); roads; lane dividers and road borders with corners; drivable-area and intersection polygons</td><td><code>autogt/lane-l3</code>, signature <code>3576fa1f7e</code>, built by the GT chat to the lane project's requests (exit events and segments were added for E273); l1 and l2 are partial leftovers of stopped runs</td></tr>
<tr><td><b>Store</b></td><td>every keyframe of all 850 trainval sequences</td><td>34,149 files, 630 MB, complete and released (tag <code>autogt/lane-l3</code>)</td></tr>
<tr><td><b>Audit</b></td><td>schema, unique ids, links inside the crop, direction rows, contiguous segments, open ends against window events, polygons, lines, finite numbers</td><td>no problem touching pieces, links, events or segments, so there is no l4; 69 stop or exit events at the window border are partial by design; 47 of 34,149 frames (0.14 %) have no lane piece</td></tr>
<tr><td><b>Release</b></td><td>the suite as one number</td><td>release v10 = front v9 + lane l3, made by one command (<code>python -m autogt generate --set full</code>)</td></tr>
<tr><td><b>Rig masks</b></td><td>static coverage of the six cameras and five radars on the lane window (0.5 m), and of the front grid</td><td>not committed yet; the six cameras together cover 99 % of the 100 x 100 m ground plane</td></tr>
</tbody></table></div>
<h3>What the next steps ask of the GT</h3>
<p>No new GT version is needed for E274: the soft node targets and the true paths are derived on the lane side from l3. The analysis of the last days does suggest two additions to discuss with the GT chat. Both are hypotheses to test before they become requests.</p>
<ul class="tight">
<li><b>Visible against inferred events.</b> Recall of nodes inside the (widened) intersection mask is {DIAG['recall_at_0_5_by']['inside intersection mask']:.2f} and of event cells {DIAG['recall_at_0_5_by']['event cell']:.2f}, against {DIAG['recall_at_0_5_by']['outside intersection mask']:.2f} outside intersections. Connector points inside intersections have no painted marker within 1 m. An attribute per event telling whether painted evidence exists nearby (stop line, crosswalk, lane divider ends) would separate events the network can learn from events it can only infer from context, for the loss weights and for the metric. Stop lines and crosswalks are in the nuScenes map expansion but not in the lane GT yet (they were left out of the first version).</li>
<li><b>What boxes hide.</b> The static camera and radar masks leave almost nothing unseen, so far-range recall ({DIAG['recall_at_0_5_by']['far (>35 m)']:.2f} beyond 35 m, against {DIAG['recall_at_0_5_by']['near (<20 m)']:.2f} below 20 m) may be about occlusion by other vehicles. A per-frame occlusion mask from the 3D boxes (rays from the cameras) would tell recall in the visible part from recall in the hidden part.</li>
<li><b>Handling of the known edge cases</b> on our side: frames without a lane piece (47) are skipped by the loss, partial window events carry a reduced weight.</li>
<li><b>Map side.</b> The lane-level map for the assignment is the lane GT itself with simulated errors (shift 1-2 m, yaw, thinning, dropped pieces); a real SD map with its own errors is a later step.</li>
</ul>"""

PIXEL = f"""
<p>The pixel stage is the first of the five. It is trained in phase 2 and stays frozen in phases 3-6; the 1 m branch starts from the E271 head. Scores below are pooled over 300 VAL frames and both seeds, a cell counts when the other mask has a cell within one cell (the tolerance of the pictures; road and intersection without tolerance).</p>
<div class="grid2"><div class="panel"><h3>Maps: precision, recall and F1 per layer</h3>{ch_pix}<p class="cap">Lane centres {pm('lane centres (1 m)', 'F1'):.2f} F1 and the area layers are strong. Lane dividers, road dividers and borders (0.5 m) reach 0.80-0.87. The intersection mask is the version widened by 2 m, as trained.</p></div>
<div class="panel"><h3>Event heat maps at 1 m</h3>{ch_pixev}<p class="cap">A predicted peak above 0.5 against a true peak above 0.5 with a tolerance of two cells. The event maps over-fire: recall {min(pm(f'event: {e} (1 m)', 'R') for e in evn):.2f}-{max(pm(f'event: {e} (1 m)', 'R') for e in evn):.2f}, precision only {min(pm(f'event: {e} (1 m)', 'P') for e in evn):.2f}-{max(pm(f'event: {e} (1 m)', 'P') for e in evn):.2f}, F1 {min(pm(f'event: {e} (1 m)', 'F1') for e in evn):.2f}-{max(pm(f'event: {e} (1 m)', 'F1') for e in evn):.2f}. The node stage then finds the events with a recall of only 0.57-0.73.</p></div></div>
<ul class="tight">
<li>Gate of phase 2 (150 frames): centre F1 at 1 m 0.910 / 0.908 against 0.917 for the E271 head (allowed: 0.02 below); road IoU 0.864 / 0.864; intersection IoU 0.770 / 0.763. Passed on both seeds.</li>
<li>The pictures below show the three bird's-eye panels of a result figure: the 1 m centre cells (dark = correct, light red = false, light blue = missed), the 2 m road and intersection with the 0.5 m dividers and borders, and the nodes with links.</li>
</ul>
<div class="grid2"><figure><a href="{pix_a}" target="_blank" rel="noopener"><img src="{pix_a}" loading="lazy" alt="pixel maps scene-0221"></a><p class="cap">scene-0221: a large junction. Centre cells and road are good; the intersection mask follows the widened polygon.</p></figure>
<figure><a href="{pix_b}" target="_blank" rel="noopener"><img src="{pix_b}" loading="lazy" alt="pixel maps scene-0092"></a><p class="cap">scene-0092: a straight road. Cells and dividers follow the lanes closely.</p></figure>
<figure><a href="{pix_c}" target="_blank" rel="noopener"><img src="{pix_c}" loading="lazy" alt="pixel maps scene-0096"></a><p class="cap">scene-0096: a busy intersection with barriers and cones.</p></figure></div>"""

SPARSE = f"""
<h3>The sparse head before E273: E271</h3>
<p>E271 put a 1 m lane-centre head on the frozen features with a differentiable sparsification and compared it with the node ids of the best graph (control r). On VAL at 10.3 vertices per 100 m the head reached coverage (cov5) 0.721 / 0.728 and precision (prec1) 0.871 / 0.873 on two seeds, against 0.591 and 0.701 for r (+0.133 coverage, interval above +0.10). The honest reading: against a budget-matched resampling of the prototype's lines the coverage gain is only about +0.03, the clear gain is precision (0.87 against 0.55-0.63). Coverage saturates near 0.74 because the head spends too few weights above the threshold floor. Cost: 3.0 ms for head and sparsification.</p>
<h3>Nodes, links, sections in E273</h3>
<div class="grid2">
<div class="panel"><h3>Nodes</h3><p>Node F1 {m2(lambda s: ND[s]['node_F1']):.3f} at 2 m (precision {m2(lambda s: ND[s]['node_P']):.2f}, recall {m2(lambda s: ND[s]['node_R']):.2f}); median position error {m2(lambda s: ND[s]['pos_err_med']):.2f} m; {m2(lambda s: ND[s]['nodes_per_100m']):.0f} nodes per 100 m. The gate (F1 not 0.05 below the E271 read-out, error up to 0.5 m) is passed with a wide margin on the cell grid (0.77 against 0.61).</p></div>
<div class="panel"><h3>Links</h3>{ch_link}<p class="cap">Short links are good, long links through intersections are not.</p></div>
<div class="panel"><h3>Sections</h3><p>{m2(lambda s: GR[s]['pieces_per_gt_piece']):.2f} delivered pieces per true piece (gate: at most 2, passed), {m2(lambda s: GR[s]['id_switches_per_100m']):.1f} id switches per 100 m, but only 47-48 % of the section starts are right (gate 0.95, failed). The soft walk needs good links; it cannot repair missing ones.</p></div>
<div class="panel"><h3>Delivered graph against the best graph so far</h3>{ch_graph}<p class="cap">Lane-suite functions on 873 VAL frames. Vertices per 100 m after Douglas-Peucker 0.3 m: {v100[0]:.1f} and {v100[1]:.1f}, against the budget 14.3; Q (without the sparsity term) {m2(lambda s: GR[s]['Q']):.3f} against {D['Q']:.3f}.</p></div></div>
<h3>The read-out threshold: what retraining must achieve, and what it cannot</h3>
<p>At the grid-cell level, with the node threshold at 0.5, the head finds {DIAG['thresholds']['0.5']['recall']:.2f} of the true node cells. {sum(v for k, v in DIAG['e_hist_of_missed'].items() if k in ('0.2-0.3', '0.3-0.4', '0.4-0.5')) / DIAG['missed_cells']:.0%} of the missed cells already have a probability between 0.2 and 0.5: the head is under-confident, it does not lack evidence. At threshold 0.3 cell recall is {DIAG['thresholds']['0.3']['recall']:.2f}, and {DIAG['thresholds']['0.3']['share_of_false_cells_touching_a_gt_cell']:.0%} of the additional false cells touch a true cell (a one-cell slip, harmless at the scorer's 2 m tolerance). Recall is worst in intersections ({DIAG['recall_at_0_5_by']['inside intersection mask']:.2f}), at event cells ({DIAG['recall_at_0_5_by']['event cell']:.2f}) and beyond 35 m ({DIAG['recall_at_0_5_by']['far (>35 m)']:.2f}). The one-cell targets are ambiguous: a lane near a cell border puts its node in one of two cells.</p>
<div class="grid2"><div class="panel"><h3>Where the true nodes are lost</h3>{ch_recall}<p class="cap">Recall of the true nodes within 2 m of a delivered node, seed 0, 300 VAL frames, exact map. Inside the grown intersection mask only {DIAG2['inside intersection']['recall_2m']:.2f} of the true nodes are found against {DIAG2['outside']['recall_2m']:.2f} outside; those {DIAG2['inside intersection']['n']:,} nodes are {DIAG2['inside intersection']['n'] / DIAG2['true_nodes']:.0%} of all nodes but about half of the misses.</p></div>
<div class="panel"><h3>The misses are gaps, not slips</h3><p>The median missed true node lies {DIAG2['median_distance_to_nearest_delivered_of_missed']:.1f} m from the nearest delivered node, and {1 - DIAG2['missed_with_delivered_within_3m']:.0%} of the missed nodes have no delivered node within 3 m. The offset of the true node from its cell centre matters only a little: recall {DIAG2['offset 0-0.5 m']['recall_2m']:.2f} when the node is within 0.5 m of the cell centre, {DIAG2['offset 1.1-1.5 m']['recall_2m']:.2f} at 1.1-1.5 m. So whole stretches are missing, mostly inside intersections and far away, where there is little painted evidence. This is why E274 weights the cells where nodes are weak instead of softening the targets.</p></div></div>
{SW_HTML}"""

ASSIGN = f"""
<p>Two experiments cover the assignment. E272 trained it inside the vertex chain but with the exact SD map, so it learned nothing beyond the geometric start value (road accuracy 0.984 trained against 0.984 start value). E273 made the task hard: a lane-level map with simulated errors, perceived nodes assigned to map pieces with a dustbin.</p>
<div class="grid2"><div class="panel"><h3>Lane accuracy</h3>{ch_assign}<p class="cap">Share of the delivered nodes within 1 m of a true lane whose most probable map piece is the map piece of that lane. Control g: the E272 chain with its geometric assigner on the same map tokens.</p></div>
<div class="panel"><h3>Numbers (mean of the two seeds)</h3><div class="scroll"><table><thead><tr><th></th><th class="num">head</th><th class="num">control g</th></tr></thead><tbody>
<tr><td>lane accuracy, exact map</td><td class="num">{m2(lambda s: P[s]['assign_exact']['lane_acc']):.3f}</td><td class="num">{m2(lambda s: G[s]['assign_exact']['lane_acc']):.3f}</td></tr>
<tr><td>lane accuracy, map 1-2 m off</td><td class="num">{m2(lambda s: P[s]['assign_error']['lane_acc']):.3f}</td><td class="num">{m2(lambda s: G[s]['assign_error']['lane_acc']):.3f}</td></tr>
<tr><td>road accuracy, map 1-2 m off</td><td class="num">{m2(lambda s: P[s]['assign_error']['road_acc']):.3f}</td><td class="num">{m2(lambda s: G[s]['assign_error']['road_acc']):.3f}</td></tr>
<tr><td>dustbin precision / recall, map 1-2 m off</td><td class="num">{m2(lambda s: P[s]['assign_error']['dust_P']):.2f} / {m2(lambda s: P[s]['assign_error']['dust_R']):.2f}</td><td class="num">{m2(lambda s: G[s]['assign_error']['dust_P']):.2f} / {m2(lambda s: G[s]['assign_error']['dust_R']):.2f}</td></tr>
<tr><td>purity, map 1-2 m off</td><td class="num">{m2(lambda s: P[s]['assign_error']['purity']):.3f}</td><td class="num">{m2(lambda s: G[s]['assign_error']['purity']):.3f}</td></tr>
</tbody></table></div><p class="cap">p - g with a wrong map: {sg(mean_err)} (95 % intervals [{b0['error']['lo']:+.3f}, {b0['error']['hi']:+.3f}] and [{b1['error']['lo']:+.3f}, {b1['error']['hi']:+.3f}]); with the exact map {sg(mean_ex)}, inside the guard.</p></div></div>
<ul class="tight">
<li><b>What limits it.</b> The assignment can only label nodes that exist and sit on a lane: node recall and the section ids are measured separately, and a node with no true counterpart must go to the dustbin (recall of that class {m2(lambda s: P[s]['assign_error']['dust_R']):.2f} against {m2(lambda s: G[s]['assign_error']['dust_R']):.2f} for geometry).</li>
<li><b>What was not tested.</b> Only two map conditions (exact, 1-2 m); no larger shifts, no sensor failure, no real SD map errors; the structure term and the registration were not ablated.</li>
<li><b>The module is independent of the graph quality:</b> it needs node tokens, positions, headings and existence; it stays as it is while nodes and links are improved.</li>
</ul>"""

LOOK = (f"""<p>Everything rendered from the networks' real predictions with the same layout: six camera views with the lane graph projected, the pixel maps with errors, the node graph with links, true piece ids against delivered ids. The comparison videos stack the best graph so far (top, red over green ground truth in the same layout) over the E273 head (bottom). 2 frames per second, 1120 px wide WebM. The numbers inside the best graph's pictures come through the E273 read-out and differ from the stored suite numbers by up to 0.01; its node F1 and the red and blue node tints are not meaningful because its vertices are not on the 2 m grid.</p>"""
        + "".join(f'<div class="scene"><h3>{n}</h3><p class="cap" style="margin-top:0">{media[n]["desc"]}</p>'
                  + (f'<figure><video controls preload="none" loop playsinline poster="{media[n]["fig"].get(0, "")}" src="{cmp_vid[n]}"></video><p class="cap">Top: best graph so far (recall_sp0_dp). Bottom: E273 head, seed 0.</p></figure>' if n in cmp_vid else "")
                  + (f'<figure><a href="{media[n]["fig"][0]}" target="_blank" rel="noopener"><img src="{media[n]["fig"][0]}" loading="lazy" alt="{n}"></a><p class="cap">E273 seed 0, middle frame, full size.</p></figure>' if media[n]["fig"].get(0) else "")
                  + (f'<figure><video controls preload="none" loop playsinline poster="{media[n]["fig"].get(1, "")}" src="{media[n]["vid"][1]}"></video><p class="cap">E273 seed 1, the whole scene.</p></figure>' if media[n]["vid"].get(1) else "")
                  + "</div>" for _, n, _ in SC)
        + (f'<h3>The perfect prediction</h3><figure><video controls preload="none" loop playsinline poster="{orc_img}" src="{orc_vid}"></video><p class="cap">The ground truth itself as the prediction (the oracle) on scene-0221: what the numbers are measured against.</p></figure>' if orc_vid else "")
        + "<h3>Earlier work</h3><div class=\"vids\">" + "".join(f'<figure><video controls preload="none" loop playsinline src="{v}"></video><p class="cap">{c}</p></figure>' for v, c in earlier) + "</div>")

WEAK = f"""
<p>Ranked by how much they cost the delivered graph today. Items 1-3 are the targets of E274.</p>
<ol class="tight">
<li><b>Connectivity at junctions.</b> Junction F1 {mj:.3f} against {D['junction_F1']:.3f}. Long links through intersections have precision {m2(lambda s: ND[s]['link_long_P']):.2f} and recall {m2(lambda s: ND[s]['link_long_R']):.2f}; every junction shows false and missed connectors.</li>
<li><b>Node recall.</b> {1 - m2(lambda s: ND[s]['node_R']):.0%} of the true nodes are missing at 2 m; the head is under-confident near the 0.5 threshold, and recall is lowest in intersections, at event cells and far away. A perfect graph with that recall would score topo F1 {drop['any_0.17']['topo_F1']:.2f}.</li>
<li><b>Event nodes.</b> Stop, exit, split, merge, start and end are found in 35-74 % of the cases with precision 0.28-0.42. The pixel event maps over-fire (high recall, low precision), so the loss sits in the node stage.</li>
<li><b>Density.</b> {min(v100):.0f}-{max(v100):.0f} vertices per 100 m after Douglas-Peucker against the budget 14.3. Node position jitter of {m2(lambda s: ND[s]['pos_err_med']):.2f} m is close to the 0.3 m tolerance, so straight lanes keep many vertices. The E271 sparsification is not on the node chain yet.</li>
<li><b>Section starts</b> only 47-48 % right.</li>
<li><b>Far range.</b> Lanes found {m2(lambda s: GR[s]['lanes_found']):.2f} against {D['lanes_found']:.2f}; the old chain's coverage-at-range gap against the slow control is also still open.</li>
<li><b>Untested parts.</b> The hold (it closes itself on clean frames), robustness to camera failure and to larger map errors, the route attribute in the new head, a closed-loop planner metric that responds.</li>
<li><b>Evidence.</b> Two seeds, VAL only, no TEST for the new head, no query-decoder reference (CGNet).</li>
</ol>"""

NEXT = f"""
<p>This is my recommendation, in the order the evidence supports. The E274 card, with its written rule, comes first; no training starts before it.</p>
<h3>E274: node recall, path-level link loss, sparsification</h3>
<ul class="tight">
<li><b>Question.</b> Can recall weights at the cells where nodes are weak (events, intersections) and a matching loss over the long links bring node recall towards 90 % and topo F1 and F1@1.0 within 0.03 of the best graph so far, while the assignment keeps its gain? The sparsification (vertex budget) is a later arm.</li>
<li><b>Arms.</b> control c = the E273 checkpoints (seeds 0 and 1); r = recall weights, trained like E273 (all phases, about 2.4 GPU hours per seed, both seeds at once); p = r's node model plus the long-link matching loss in phases 4-7 (about 1.3 GPU hours per seed); later s = sparsification. The card is <code>experiments/E274_node-recall-and-long-link-matching/experiment.md</code>; the code switches are tested and under a critic pass.</li>
<li><b>Node recall changes (arm r).</b> The measurements changed the design: the missing nodes are gaps, not one-cell slips (the median missed node is 3.8 m from the nearest delivered node), and 55 % of them lie inside intersections (recall 0.73 against 0.94 outside). So r weights the positives of the existence loss at event cells (x3) and inside intersections (x2). The pixel event maps over-fire (recall high, precision low), so their loss stays as it is; an earlier draft tripled it and the critic pass showed that the evidence does not support that. The read-out stays at 0.5.</li>
<li><b>Long-link matching (arm p).</b> The path-level loss on chains of short links is dropped: the product of link probabilities along a path is the sum of the per-link losses, and the oracle probe says chain breaks are mostly node gaps. What is left is the junction, an entry-to-exit assignment: the long-link candidates of one entry compete, so do those of one exit, each with a none entry (dual softmax, SuperGlue style). The all-link row softmax of E273 (attempt v6) was worse; this version touches only the long links.</li>
<li><b>Written rule.</b> r against the E273 checkpoints: KEEP if topo F1 rises by more than 0.02 on both seeds with the guards held (assignment gain +0.03 over the geometric control kept, exact map within 0.01, F1@1.0 not below the control by more than 0.01); TIE inside +-0.02; CUT below. p against r on junction F1 with band 0.03. Reported, not judged: the system pass (topo F1 and F1@1.0 within 0.03 of the best graph so far), density, duplicates.</li>
<li><b>Diagnostics done for it.</b> Cell-level node misses, the read-out threshold sweep, the per-layer pixel scores, and the oracle with nodes removed (recall explains the topology gap, the links explain the junction gap). They are part of the card.</li>
</ul>
<h3>What I need from you</h3>
<ul class="tight">
<li>Nothing blocks the start: I follow the card as written. Tell me if you want the weights or the arms changed before the training starts at 08:30.</li>
<li>Your view on the two GT ideas (visible against inferred events, occlusion by boxes): I will raise them with the GT chat only after you say so.</li>
<li>A TEST check is not planned yet; it needs your recorded OK once a design is finished.</li>
</ul>
<h3>After E274</h3>
<ol class="tight">
<li>Temporal fusion on the nodes by cross-attention from the current nodes to the nodes of earlier frames (your idea), with a degraded-frame evaluation so the hold's value can be seen.</li>
<li>The route attribute and the planner consumer on the new head.</li>
<li>A query-decoder reference (CGNet) scored with the same lane suite as arm Q of the E273 card.</li>
<li>Robustness: larger map errors, sensor failure.</li>
</ol>"""
