p = "evaluation/bev_selftest.py"
s = open(p, encoding="utf-8").read()
anchor = '@check("taps of the new modules:'
assert s.count(anchor) == 1
new = '''@check("palette of the bird's-eye pictures: black background, dark-grey road with a black boundary, white lane lines, light-grey crossings, dark-red objects in the observed area and "
       "the same red translucent outside it; observed_area = camera x radar x not hidden; box outlines by the observed area")
def _():
    from evaluation import bev_views as V
    n = 12
    z = np.zeros((n, n), np.float32)
    road, cross, veh = z.copy(), z.copy(), z.copy()
    road[3:9, 3:9], cross[5, 5], veh[4, 4], veh[4, 8] = 1, 1, 1, 1
    obs = np.ones((n, n), bool)
    obs[:, 7:] = False                                                          # columns >= 7 are outside the observed area
    line = np.full((n, n), 9.0, np.float32)
    line[:, 1] = 0.0                                                            # a lane divider along column 1, on the background
    img = V.composite({"drivable_area": road, "ped_crossing": cross, "vehicle": veh}, {"lane_divider": line}, observed=obs)
    close = lambda a, b: bool(np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-5)       # noqa: E731
    assert close(img[0, 5], V.BG) and close(img[6, 6], V.COLORS["drivable_area"]), "black background, dark-grey road"
    assert close(img[3, 5], V.ROAD_EDGE) and close(img[6, 3], V.ROAD_EDGE), "the road boundary is black"
    assert close(img[5, 5], V.COLORS["ped_crossing"]) and close(img[0, 1], V.LINE_COLORS["lane_divider"]) and V.COLORS["ped_crossing"][0] > 0.6 > V.COLORS["drivable_area"][0], "crossing light grey, lane line white"
    assert close(img[4, 4], V.DARK_RED), "an object in the observed area is dark red"
    assert close(img[4, 8], np.asarray(V.OUT_RED) * V.OUT_ALPHA + np.asarray(V.ROAD_EDGE) * (1 - V.OUT_ALPHA)), "outside it: the same red, translucent, over what is below"
    plain = V.composite({"drivable_area": road, "vehicle": veh}, {})
    assert close(plain[4, 8], V.DARK_RED), "without an observed area every object is dark red"
    g = {"camera": np.ones((4, 4)), "occluded": np.zeros((4, 4)), "radar": np.ones((4, 4))}
    g["camera"][:, 0], g["occluded"][1, 1], g["radar"][2, 2] = 0, 1, 0
    a = V.observed_area(g)
    assert a.sum() == 16 - 4 - 1 - 1 + 0 and not a[:, 0].any() and not a[1, 1] and not a[2, 2] and a[0, 1], "camera x radar x not hidden"
    assert V.observed_area({"camera": np.ones((4, 4))}).all() and V.observed_area({"undefined": np.zeros((3, 5))}).shape == (3, 5)
    area = np.zeros((120, 104), bool)
    area[:20] = True                                                           # x < 10 m observed
    inside, outside = dict(cx=5.0, cy=0.0, length=4.0, width=2.0, yaw=0.0), dict(cx=30.0, cy=0.0, length=4.0, width=2.0, yaw=0.0)
    cols = [sh.color for sh in V.box_shapes([inside, outside], V.GT_BOX_COLOR, observed=area) if sh.kind == "polygon"]
    assert cols == [V.GT_BOX_COLOR, V.GT_BOX_OUT] and V.GT_BOX_OUT[3] < 1.0, cols
    assert {sh.color for sh in V.box_shapes([inside, outside], "w")} == {"w"}, "without an observed area one colour"


'''
s = s.replace(anchor, new + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
