"""E363 scratch: a run folder of the pair run of E324 s whose NODE HEAD is the one of B' (E292 c2 or a): `runs/e363_<g>_s<S>/model.pt` = the checkpoint of `runs/e324_s_s<S>` with every `nodes.*` tensor replaced by the `nodes.*` tensor of `runs/e292_<g>_s<S>/model.pt`
(B' = B fine-tuned with the pyramid frozen, so the pyramid and the maps head are the pair run's own and equal B''s: checked), nothing else changed. `jends_eval.py dump --run e363_<g>_s<S>` then gives the attached ends (and the keypoints) computed on B''s nodes: the FAITHFUL
dump that E360 lacked (E360 read stored dumps whose attached ends were computed on B's nodes). Nothing in a stored run is touched; an existing target folder is refused.
usage (problem folder): wsl bash tools/env/_py14.sh temp/e363/make_run.py <c2|a> <seed>"""
import hashlib
import json
import os
import sys

import torch

ROOT = os.environ["BFLC_ROOT"]
g, seed = sys.argv[1], int(sys.argv[2])
assert g in ("c2", "a", "b") and seed in (0, 1), (g, seed)                                 # b = B itself (no change of the nodes): the FIXTURE, its dump must reproduce the stored dump of E324 s
src, bp = f"{ROOT}/runs/e324_s_s{seed}/model.pt", (f"{ROOT}/runs/e273_p_s{seed}/model.pt" if g == "b" else f"{ROOT}/runs/e292_{g}_s{seed}/model.pt")
dst_dir = f"{ROOT}/runs/e363_{g}_s{seed}"
assert not os.path.exists(dst_dir), f"{dst_dir} exists: refused"
ck = torch.load(src, map_location="cpu")
bs = torch.load(bp, map_location="cpu")["state"]
nodes = {k: v for k, v in bs.items() if k.startswith("nodes.")}
mine = {k: v for k, v in ck["state"].items() if k.startswith("nodes.")}
assert nodes and set(nodes) == set(mine), f"node-head keys differ: {set(nodes) ^ set(mine)}"
assert all(nodes[k].shape == mine[k].shape for k in nodes), "node-head shapes differ"
pyr = [k for k in bs if k.startswith("pyr.")]
pyr_pair = [("jmodel." + k) if ("jmodel." + k) in ck["state"] else k for k in pyr]
assert pyr and all(torch.equal(bs[k], ck["state"][kp]) for k, kp in zip(pyr, pyr_pair)), "the pyramid of B' is not the pair run's: the connectors would sit on other pixel features"
changed = [k for k in nodes if not torch.equal(nodes[k], mine[k])]
new = dict(ck)
new["state"] = dict(ck["state"])
new["state"].update(nodes)
os.makedirs(dst_dir)
torch.save(new, f"{dst_dir}/model.pt")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
json.dump(dict(source_pair_run=f"e324_s_s{seed}", node_head_from=(f"e273_p_s{seed}" if g == "b" else f"e292_{g}_s{seed}"), node_tensors=len(nodes), node_tensors_changed=len(changed), pyramid_tensors_checked_equal=len(pyr), sha256_source=sha(src), sha256_node_head_source=sha(bp), sha256_new=sha(f"{dst_dir}/model.pt"),
               note="E363: the pair run of E324 s with the node head of B' (checkpoint surgery: only `nodes.*`); the stored runs are untouched"), open(f"{dst_dir}/provenance.json", "w"), indent=1)
print(f"{dst_dir}: {len(nodes)} node-head tensors, {len(changed)} differ from B's; {len(pyr)} pyramid tensors equal")
