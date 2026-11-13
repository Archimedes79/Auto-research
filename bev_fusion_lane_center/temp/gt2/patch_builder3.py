p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
marker = "exec(open(os.path.join(HERE, \"report_text.py\"), encoding=\"utf-8\").read())"
funcs = '''def errors_block():
    """the failure figures of temp/jr/fail_gallery.py / maneuver.py (E308 s0, p >= 0.5) and the error videos"""
    ed = os.path.join(HERE, "media", "errors")
    caps = [("distribution", "Distribution over the 873 VAL frames: coverage of the GT lanes by the delivered graph (left), true pairs delivered per frame (middle), pair probability of true and false candidates (right; log scale)."),
            ("maneuver_e308_t_s0", "Maneuver coverage: how many GT entry-to-exit maneuvers are drawn in the delivered graph, plain graph, connectors at three thresholds, and a perfect pair classifier on the same keypoints."),
            ("random12", "Twelve random junction frames: the delivered graph against the GT (green near GT, red with no GT, blue GT not covered)."),
            ("misses", "Largest junctions, most missed pairs. Left GT lanes with ends; middle the delivered graph against the GT; right the pairs (green true, red false, blue dashed missed)."),
            ("worst_coverage", "Lowest coverage: whole arms of the junction are missing from the delivered graph."),
            ("false_pairs", "Most false connectors: hairpin curves where no junction is."),
            ("typical", "A typical frame: the main road is right, the cross street is drawn only in part, two of 16 pairs are delivered.")]
    out = []
    for n, c in caps:
        pth = os.path.join(ed, n + ".png")
        if os.path.exists(pth):
            out.append(fig(jpg(pth, "err_" + n + ".jpg", maxw=1800), c))
    for pth in sorted(glob.glob(os.path.join(ed, "*_errors.webm"))):
        n = os.path.basename(pth)
        shutil.copy(pth, os.path.join(VID, n))
        out.append(fig("vid/" + n, "Video: " + n.replace("_errors.webm", "") + ", three panels per keyframe: GT lanes with ends, delivered graph against GT, pairs.", vid=True))
    return "".join(out)


'''
assert marker in s
s = s.replace(marker, funcs + marker, 1)
old = '<section id="look"><h2>Pictures and videos</h2>'
assert old in s
s = s.replace(old, '<section id="fail"><h2>Where it fails, and what the numbers say about it</h2>{FAIL}{errors_block()}</section>\n' + old, 1)
s = s.replace('<a href="#prune">Pruning, stability, depth</a>', '<a href="#prune">Pruning, stability, depth</a><a href="#fail">Where it fails</a>', 1)
if "import shutil" not in s:
    s = s.replace("import glob\n", "import glob\nimport shutil\n", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
