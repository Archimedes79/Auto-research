import json, os
p = os.path.join(os.environ["BFLC_ROOT"], "results", "E2E_Champion_Report", "report.json")
r = json.load(open(p, encoding="utf-8"))
r["teaser"]["caption"] = ("One real VAL frame (a T-junction): the plain lane head, the champion (the token head) and, for context, the previous system E324 s with rule steps, against the ground truth. "
                          "Green dots lie within 1 m of the truth, blue is truth that nothing covers. The champion covers the long lane and the crossing arms that the plain head misses (topo F1 0.71 against 0.21) "
                          "and finds 9 of the 18 maneuvers against 3; this is the frame where it gains most, not a typical one (chapter 6).")
json.dump(r, open(p, "w", encoding="utf-8"), indent=1)
print("ok")
