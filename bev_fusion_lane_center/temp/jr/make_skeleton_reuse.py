"""Scratch (run once): create the report folder of the E2E champion (D113) and reuse parts of the sent report:
01_gt.md (whole, with a E2E-view block), the data / metrics sections of 09_rebuild.md, the pictures of the GT figure."""
import os, re, shutil
R = os.environ["BFLC_ROOT"]
SRC = os.path.join(R, "results", "Ergebnis_Spurmitte_2026-11-15", "parts")
DST = os.path.join(R, "results", "Ergebnis_E2E_Champion_2026-11-17", "parts")
os.makedirs(os.path.join(DST, "04_images"), exist_ok=True)

# ---- 01_gt.md: reuse, with the E2E view in front of 1.1
t = open(os.path.join(SRC, "01_gt.md"), encoding="utf-8").read()
head_end = t.index("### 1.1")
e2e = """**The E2E view (what the token champion reads from this GT).** The champion's head sees the predicted nodes of the frozen lane head B' (2 m grid) as tokens and is trained against the same ground truth, in three forms (E365 card, section 1): (a) the GT node table on the **2.0 m lattice** (position, heading, flags: split / merge / stop / exit / start / end, piece, links with their chord length and fitted handle scales); (b) a **static assignment** of tokens to GT nodes (greedy by descending existence of B', nearest unassigned GT node within 1.5 m and heading cosine >= 0.5), which gives the node-type label (7 classes: none, lane point, fork, merge, junction entry, junction exit, boundary) and the existence target; (c) the **link targets**: for a source token the positives are every token within 1.5 m of a GT successor (the GT links are first contracted over GT nodes that no token covers), separately for short (chord <= 8 m) and long (> 8 m, junction) links. Matching inside the loss is allowed (D55). Nothing of this table is read at inference: the read-out writer takes positions and headings from the tokens only (a mutant test puts a GT field into the prediction and must fail). PENDING: the numbers of the final GT version used by the champion (signature) and the share of GT links the token set can reach (E365 manifest, VAL, B' seed 0: 94.3 % of the GT links have both ends covered and lie inside the admissible-pair gate; GT nodes assigned 0.952).

"""
t = t[:head_end] + e2e + t[head_end:]
open(os.path.join(DST, "01_gt.md"), "w", encoding="utf-8").write(t)

# ---- 09_rebuild.md: only the data and metrics sections are reused
r = open(os.path.join(SRC, "09_rebuild.md"), encoding="utf-8").read()
L = r.split("\n")
def sec(prefix):
    i = next(k for k, l in enumerate(L) if l.startswith(prefix))
    j = next((k for k in range(i + 1, len(L)) if L[k].startswith("### ")), len(L))
    return "\n".join(L[i:j]).rstrip() + "\n"
out = ["## 7. Rebuilding it: from nuScenes to the numbers", "",
       "This chapter is reused in part from the report of 2026-11-16 (data, splits, metrics); the network, the training recipe and the numbers to hit are written for the champion when it exists.", "",
       sec("### 8.1").replace("### 8.1", "### 7.1"), sec("### 8.6").replace("### 8.6", "### 7.2"),
       "### 7.3 The network, the training recipe and the numbers to hit at each stage", "",
       "PENDING: modules and shapes of the token head (chapter 2), the training recipe with commands (E365 / the champion's cards), the losses (per-source likelihood with a none slot, node-type cross-entropy 0.5, 0.5 x handle-scale L1), the fixed settings (K = 512, decode thresholds), and the numbers to hit per stage (cache fixture, layer-1 reproduction, parity, S1, the champion).", "",
       "### 7.4 Where to look", "",
       "| experiment | question | status | folder |", "|---|---|---|---|",
       "| E365 | the end-to-end token champion, stage S1 on the predicted nodes of the frozen B' | MISS by the written gate (section 3) | `E365_e2e-token-champion-s1` |",
       "| E366 | the same two heads with a decode re-selected on the TRAIN sample | PENDING | PENDING |",
       "| E367 | a wider and deeper head, token-drop training, 20000 steps | PENDING | PENDING |",
       "| E357 / E359 / E361 / E362 | the oracle-node stage of the same design (given ground-truth nodes) | INCONCLUSIVE at the end (E362) | `E357_*`, `E359_*`, `E361_*`, `E362_*` |", ""]
open(os.path.join(DST, "09_rebuild.md"), "w", encoding="utf-8").write("\n".join(out))
print("ok")
