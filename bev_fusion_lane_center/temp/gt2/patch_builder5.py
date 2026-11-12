p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
a = s.index("def errors_block():")
b = s.index("exec(open(os.path.join(HERE, \"report_text.py\")")
# keep snap_table / thr_table / common_table definitions that sit between: only replace errors_block itself
end_err = s.index("def snap_table():") if s.index("def snap_table()") > a else None
i_next = min([x for x in (s.find("\ndef ", a + 10),) if x > 0])
new_block = '''def errors_block(dirname="errors", prefix="err_", label="", maneuver=True):
    """the failure figures of temp/jr/fail_gallery.py (and maneuver coverage) and the error videos of one read-out; dirname = temp/jr/media/<dirname>"""
    ed = os.path.join(HERE, "media", dirname)
    caps = [("distribution", f"{label}Distribution over the 873 VAL frames: coverage of the GT lanes by the delivered graph (left), true pairs delivered per frame (middle), pair probability of true and false candidates (right; log scale)."),
            ("random12", f"{label}Twelve random junction frames: the delivered graph against the GT (green near GT, red with no GT, blue GT not covered)."),
            ("misses", f"{label}Largest junctions, most missed pairs. Left GT lanes with ends; middle the delivered graph against the GT; right the pairs (green true, red false, blue dashed missed)."),
            ("worst_coverage", f"{label}Lowest coverage: whole arms of the junction are missing from the delivered graph."),
            ("false_pairs", f"{label}Most false connectors."),
            ("typical", f"{label}A typical junction frame.")]
    out = []
    for n, c in caps:
        pth = os.path.join(ed, n + ".png")
        if os.path.exists(pth):
            out.append(fig(jpg(pth, prefix + n + ".jpg", maxw=1800), c))
    for pth in sorted(glob.glob(os.path.join(ed, "*_errors.webm"))):
        n = os.path.basename(pth)
        shutil.copy(pth, os.path.join(VID, prefix + n))
        out.append(fig("vid/" + prefix + n, f"{label}Video: " + n.replace("_errors.webm", "") + ", three panels per keyframe: GT lanes with ends, delivered graph against GT, pairs.", vid=True))
    return "".join(out)


'''
s = s[:a] + new_block + s[i_next + 1:]
s = s.replace("{FAIL}{errors_block()}", "{FAIL}<h3>After: E305 pairs, connector ends snapped, p >= 0.1</h3>{errors_block('errors_snap', 'aft_', 'After (E305, snapped, p >= 0.1). ')}<h3>Before: E308 pairs, connectors from the keypoints, p >= 0.5</h3>{errors_block('errors', 'err_', 'Before (E308, p >= 0.5). ')}")
open(p, "w", encoding="utf-8").write(s)
print("ok")
