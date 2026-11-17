import json, os
p = os.path.join(os.environ["BFLC_ROOT"], "results", "E2E_Champion_Report", "report.json")
r = json.load(open(p, encoding="utf-8"))
r["pack"] = {"limit": 2000000, "code": ["../../../winners/e365_token_champion/**/*"], "strip": "../../../winners/e365_token_champion", "require_clean_git": False,
 "readme": ("E2E token champion (D114): the paper-style report (report.html, open in any browser) and the champion's code.\n\n"
            "code/ = winners/e365_token_champion: the token head (training/grid/jtoken.py), the loss and decode, the loader and token cache, the trainer, the evaluation and their tests, "
            "with repository-relative paths (copy code/code/ over bev_fusion_lane_center/; it needs the repository modules named in code/README.md). MANIFEST.json lists the files with sha256 and the source commit.\n"
            "utilities/ = the helper scripts that stay with this work (the report builder, the chart tool, the winner exporter).\n"
            "The weights are not in this package (size limit): the heads are runs/e365_s1_s0, runs/e365_s1_s1, runs/e371_cf_s0, runs/e371_cf_s1 and the frozen lane head runs/e292_a_s0 / s1 in the repository.\n"
            "Status: stage S1 with the lane head frozen; no TEST number; a challenger (E373) is under confirmation.\n")}
json.dump(r, open(p, "w", encoding="utf-8"), indent=1)
print("ok")
