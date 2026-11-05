"""scratch: step time, data wait and GPU memory of the three large runs from runs/<run>/log.jsonl (the rows with a 't_step' or 's_per_step' key)
    wsl python3 bev_fusion/temp/t16/train_stats.py"""
import json
import statistics as st

for r in ("t16_full_s0", "t17_data", "t18_fix"):
    rows = [json.loads(l) for l in open(f"bev_fusion/runs/{r}/log.jsonl", encoding="utf-8") if l.strip()]
    keys = {}
    for x in rows:
        for k, v in x.items():
            if isinstance(v, (int, float)):
                keys.setdefault(k, []).append(v)
    ts = keys.get("t_step") or []
    sps = keys.get("s_per_step") or []
    td = keys.get("t_data") or []
    gg = keys.get("gpu_gb") or []
    print(r, "rows", len(rows), "| t_step n", len(ts), "median", round(st.median(ts), 2) if ts else None, "mean", round(st.mean(ts), 2) if ts else None,
          "| s_per_step n", len(sps), "median", round(st.median(sps), 2) if sps else None, "mean", round(st.mean(sps), 2) if sps else None,
          "| t_data share", round(sum(td) / sum(ts), 3) if td and ts and len(td) == len(ts) else (round(st.mean(td), 3) if td else None), "| gpu_gb max", max(gg) if gg else None)
