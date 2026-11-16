"""one-off: splice the generated table into the card's section 10 and the result READMEs, and write the per-run READMEs from the scoring cards"""
import json
import subprocess

table = open("temp/e273/table.md", encoding="utf-8").read().split("\np - g with the 95 %")[0].rstrip()
boot = "p - g with the 95 %" + open("temp/e273/table.md", encoding="utf-8").read().split("\np - g with the 95 %")[1]
table = table + "\n\n" + boot.rstrip()

# card section 10
card = "experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md"
s = open(card, encoding="utf-8").read()
head, _, tail = s.partition("## 10. Summary\n")
assert tail.strip() == "Open.", tail[:80]
s10 = open("temp/e273/card_s10.md", encoding="utf-8").read().replace("@@TABLE@@", table)
open(card, "w", encoding="utf-8").write(head + s10)

# index README
idx = open("temp/e273/readme_index.md", encoding="utf-8").read().replace("@@TABLE@@", table)
import os
os.makedirs("results/E273", exist_ok=True)
open("results/E273/README.md", "w", encoding="utf-8").write(idx)

# per-run READMEs
for seed in (0, 1):
    run, rid = f"e273_p_s{seed}", f"E273.p_s{seed}"
    c = json.load(open(f"runs/e273/{run}/card.json"))
    pk = json.load(open(f"results/{rid}/package.json"))
    tm = json.load(open(f"runs/e273/{run}/timing.json"))
    g, p = c["p"]["graph"], c["p"]
    ae, ax = p["assign_error"], p["assign_exact"]
    d = c["diff"]
    txt = f"""# {rid}: E273 lane head, seed {seed}

Run of the experiment index `../E273/README.md` (card `experiments/E273_pyramid-head-soft-graph-map-assignment/experiment.md`). Verdict of the experiment: CUT by hand (both seeds, rule 4 of the card).

| | |
|---|---|
| ledger row | `{rid}` (status {pk['status']['status']}) |
| code | tag `{pk['code']['tag']}`, run commit `{pk['code']['commit']}` (the launcher's snapshot of the head code, `training/grid/`) |
| checkpoint | `runs/{run}/model.pt`, hash `{pk['model']['ckpt_hash']}` (not in git) |
| lane GT | `{pk['gt']['version']}`, signature `{pk['gt']['signature']}`; front GT of the cached features `c643a7539d` |
| evaluation | VAL {pk['evaluation']['frames']} frames, frame hash `{pk['evaluation']['frame_hash']}`; evaluator commit `{pk['evaluation']['evaluator']['commit']}`, code hash `{pk['evaluation']['evaluator']['code_hash']}` |
| scoring card | `runs/e273/{run}/card.json`, `results.md` (rebuild: `wsl bash tools/env/_py14.sh qgraph_run_eval.py score --run {run} --seed {seed}`) |

## Numbers (VAL {c['frames']} frames, from the scoring card)
- lane accuracy of the assignment, map error 1-2 m: p {ae['lane_acc']:.4f}, control g {c['g']['assign_error']['lane_acc']:.4f}, p - g {d['error']['mean']:+.4f} [{d['error']['lo']:+.4f}, {d['error']['hi']:+.4f}]
- lane accuracy, exact map: p {ax['lane_acc']:.4f}, g {c['g']['assign_exact']['lane_acc']:.4f}, p - g {d['exact']['mean']:+.4f} [{d['exact']['lo']:+.4f}, {d['exact']['hi']:+.4f}]
- delivered graph (lane suite functions): topo F1 {g['topo_F1']:.3f}, F1@1.0 {g['F1@1.0']:.3f}, junction F1 {g['junction_F1']:.3f}, lanes found {g['lanes_found']:.3f}, Q {g['Q']:.3f}, duplicates {g['dup_frac']:.3f}; reference d: 0.741 / 0.878 / 0.273 / 0.781 / 0.742 / 0.077
- nodes: F1 {p['nodes']['node_F1']:.3f} (P {p['nodes']['node_P']:.3f}, R {p['nodes']['node_R']:.3f}), median position error {p['nodes']['pos_err_med']:.3f} m; vertices per 100 m {g['vert100']:.1f} node ids / {g['vert100_dp']:.1f} after Douglas-Peucker 0.3 m; pieces per GT piece {g['pieces_per_gt_piece']:.2f}
- time per frame, head only (idle GPU, batch 1, bf16): {tm['total']['median']:.1f} ms = pixel maps {tm['pyr']['median']:.1f} + nodes {tm['nodes']['median']:.1f} + links {tm['links']['median']:.1f} + sections {tm['sections']['median']:.1f} + map and assignment {tm['assign']['median']:.1f}

## Files
- `figures/`: one frame per scene (camera views, pixel maps with errors, node graph, ids); `videos/`: the three scenes as mp4; `frames/<scene index>/`: every frame of those scenes; not in git
- `package.json`, `provenance.json`: the result package (versions, hashes, the numbers); fields the generic packager could not read were filled from the run's records
- rebuild pictures: `wsl bash tools/env/_py14.sh qgraph_results.py --run {run} --id {rid} --device cpu --scenes 11 39 23`
"""
    open(f"results/{rid}/README.md", "w", encoding="utf-8").write(txt)
print("ok")
