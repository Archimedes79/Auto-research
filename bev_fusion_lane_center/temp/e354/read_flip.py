import json, sys
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "missing"); continue
    c, r, m = d["control"], d["run"], d["run_matched"]
    print(f.split("/")[1], "vs", d["control_name"], "| control: delivered", c["true_pairs_t0_delivered"], "flip", round(c["flip_rate"], 4),
          "| run: delivered", r["true_pairs_t0_delivered"], "flip", round(r["flip_rate"], 4),
          "| run matched (thr %.3f): delivered %d flip %.4f" % (m["thr"], m["true_pairs_t0_delivered"], m["flip_rate"]))
    for k in ("equal_threshold", "matched_delivery"):
        e = d[k]; print("   ", k, "diff %+.4f [%+.4f, %+.4f]" % (e["difference"], e["lo95"], e["hi95"]))
