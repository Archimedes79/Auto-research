import json, sys
import numpy as np
for d in sys.argv[1:]:
    r = json.load(open(f"../jr/media/{d}/stats.json"))
    J = [x for x in r if x["n_gt_pairs"] >= 4]
    fr = [x["n_true"] / x["n_gt_pairs"] for x in J]
    print(d, "junction frames", len(J), "mean pair recall %.3f" % np.mean(fr), "frames none", sum(1 for f in fr if f == 0), "coverage plain %.3f head %.3f precision %.3f" % (np.mean([x["recall_base"] for x in J]), np.mean([x["recall_head"] for x in J]), np.mean([x["precision_head"] for x in J])))
