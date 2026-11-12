import json, os
r = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "references.json")))
print("| row | frames | nodes per 100 m | free recall | free exact | fixed recall | off-zone | all-zone recall / exact |")
print("|---|---|---|---|---|---|---|---|")
for x in r["rows"]:
    print("| %s | %d | %.1f | %.3f | %.3f | %.3f | %.3f | %.3f / %.3f |" % (x["name"], x["frames"], x["nodes_per_100m"], x["free_zone_recall"], x["free_zone_exact"], x["fixed_zone_recall"], x["off_zone"], x["zone_recall"], x["zone_exact"]))
print(r["meta"])
