"""Scratch (run once): turn the skeleton parts of results/Ergebnis_E2E_Champion_2026-11-17 into a reporter folder in paper order
(results/E2E_Champion_Report/). Reused parts are copied and renumbered; the new chapters are written by hand (the Write tool) before this runs."""
import os, re, shutil
R = os.environ["BFLC_ROOT"]
SK = os.path.join(R, "results", "Ergebnis_E2E_Champion_2026-11-17", "parts")
OLD = os.path.join(R, "results", "Ergebnis_Spurmitte_2026-11-15", "parts")
NEW = os.path.join(R, "results", "E2E_Champion_Report", "parts")
os.makedirs(NEW, exist_ok=True)
os.makedirs(os.path.join(NEW, "figures"), exist_ok=True)
rd = lambda p: open(p, encoding="utf-8").read()


def wr(name, text):
    open(os.path.join(NEW, name), "w", encoding="utf-8").write(text)


# figures
for f in ("00_gt_junction_layer.jpg", "03_junction_win_E324s_largest_gain.jpg", "04_far_lanes_missed.jpg", "05_junction_maneuvers_missing.jpg"):
    shutil.copy(os.path.join(OLD, "04_images", f), os.path.join(NEW, "figures", f))

# 2 system: the architecture chapter of the skeleton
shutil.copy(os.path.join(SK, "02_architecture.svg"), os.path.join(NEW, "02_system.svg"))
wr("02_system.md", rd(os.path.join(SK, "02_architecture.md")).replace("## 2. System architecture: the end-to-end token head", "# 2. The system: architecture and description", 1))

# 3 GT: the reused chapter, renumbered, with its picture
gt = rd(os.path.join(SK, "01_gt.md"))
gt = gt.replace("](04_images/", "](parts/figures/")
gt = re.sub(r"^### 1\.(\d)", r"### 3.\1", gt, flags=re.M)
gt = re.sub(r"^## 1\. ", "# 3. ", gt, flags=re.M)
SHORT = ("**In short.** nuScenes provides the HD map, the recorded poses and the camera and radar data, but no lane-graph labels. A generator derives them: "
         "directed centre lines with split, merge, stop and exit events, which lane the car really drives, what the cameras can see, and for junctions the zone, "
         "the arms and the valid entry-to-exit pairs. The same graph code makes the training targets and the scoring reference, so a number is always measured "
         "against the label it was trained on. The token head additionally needs the 2.0 m lattice table, a static assignment of tokens to GT nodes and the link "
         "targets (below). The picture shows the labels of one junction frame; the details, the checks and the limits follow.")
H = "# 3. Required ground truth\n"
gt = re.sub(r"^# 3\. [^\n]*\n", H, gt, count=1, flags=re.M)
assert gt.count(H) == 1
gt = gt.replace(H, H + "\n" + SHORT + "\n", 1)
wr("03_gt.md", gt)

# A repeat: data + metrics reused from the skeleton
rb = rd(os.path.join(SK, "09_rebuild.md"))
rb = rb.replace("## 7. Rebuilding it: from nuScenes to the numbers", "# A. How to repeat the result", 1)
rb = re.sub(r"^### 7\.(\d)", r"### A.\1", rb, flags=re.M)
wr("A1_repeat.md", rb)

# 5 quantitative: the results skeleton, with numbered table captions
q = rd(os.path.join(SK, "03_results.md"))
q = q.replace("# Results: the end-to-end token champion (skeleton)", "# 5. Quantitative results", 1)
caps = [("| step | system | topo F1 |", "Table: The ladder of end-to-end systems of this project (topo F1, VAL 873 frames, seed 0 / seed 1). Each step removes or replaces a hard step; rule-step systems are not in the table."),
        ("| condition | floor (s0 / s1) | S1 | holds? |", "Table: The written gate for an \"okayish\" champion (D113), per seed, and what stage S1 reached."),
        ("| VAL 873, delivered graph, every section 0 |", "Table: Stage S1 against its controls: plain B, the same nodes and links read with sections 0, a one-line rule on the same tokens, the ceiling of the token set (GT links over the assigned tokens), and an untrained head."),
        ("| system | parameters (M) | ms per frame | note |", "Table: Compute of the reference, of the parts of S1 and (PENDING) of the champion; the ms of S1 were measured on a host that was not quiet.")]
for head, cap in caps:
    assert q.count(head) == 1, head
    q = q.replace(head, cap + "\n\n" + head)
HA = "## A. The ladder: what each end-to-end step gave (topo F1, VAL, s0 / s1)\n"
FIG = ("![The ladder of end-to-end systems and the token head, topo F1 on VAL for both seeds. The dashed line is the written floor of the gate "
       "(plain lane head + 0.01). Look at the token head S1 (left of the line) against the earlier systems; the hatched rows are not measured yet (PENDING). The axis starts at 0.5.]"
       "(parts/figures/ladder.png)")
assert q.count(HA) == 1, "ladder heading"
q = q.replace(HA, HA + "\n" + FIG + "\n", 1)
wr("05_quant.md", q)
print("ok")
