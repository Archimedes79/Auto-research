set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
a = '''    ok = res["checks"]
    ok["points warp within tolerance"] = point_pass(res["points"]["warp"])
    ok["headings warp p90 within tolerance"] = res["headings"]["warp"]["n"] > 0 and res["headings"]["warp"]["p90"] <= TOL["head_p90_deg"]
    ok["tokens warp p90 within tolerance"] = res["tokens"]["warp"]["n"] > 0 and res["tokens"]["warp"]["p90"] <= TOL["tok_p90"]
    for how in hows:
        if how != "warp":
            ok[f"{how} fails the point test"] = not point_pass(res["points"][how])
    res["passed"] = all(ok.values())
'''
b = '''    ok = res["checks"] = fixture_checks(res["points"], res["headings"], res["tokens"], hows)
    res["passed"] = all(ok.values())
'''
assert s.count(a) == 1
s = s.replace(a, b)
a2 = "def fixture(split="
b2 = '''def fixture_checks(points, headings, tokens, hows=CONVENTIONS, tol=TOL):
    """the rule of the fixture on the stats per convention (dicts of `stats`) -> name -> bool: the right pose within every tolerance, every wrong convention FAILS the point test"""
    ok = {"points warp within tolerance": point_pass(points["warp"], tol),
          "headings warp p90 within tolerance": headings["warp"]["n"] > 0 and headings["warp"]["p90"] <= tol["head_p90_deg"],
          "tokens warp p90 within tolerance": tokens["warp"]["n"] > 0 and tokens["warp"]["p90"] <= tol["tok_p90"]}
    for how in hows:
        if how != "warp":
            ok[f"{how} fails the point test"] = not point_pass(points[how], tol)
    return ok


def fixture(split='''
assert s.count(a2) == 1
s = s.replace(a2, b2)
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
