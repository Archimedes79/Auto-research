import json, sys
O = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs"
for n in sys.argv[1:]:
    try:
        d = json.load(open(f"{O}/{n}/metrics.json"))
        bt = max(d["val"].items(), key=lambda kv: kv[1]["F1@1.0"])[0]
        extra = ""
        for k in ("val_align", "test_align"):
            if k in d:
                extra += f" {k}: err {d[k]['pos_err_median']:.2f} (before {d[k]['pos_err_before_median']:.2f})"
        print(n, "VAL %.3f TEST %.3f (tau %s)" % (d["val"][bt]["F1@1.0"], d["test"][bt]["F1@1.0"], bt), extra)
    except Exception as e:
        print(n, "ERR", e)
