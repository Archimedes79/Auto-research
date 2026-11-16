import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_data.py", [
("""att_dir=torch.full((2, 2, 2), 0.5))""", """att_dir=0.5 + torch.arange(8.0).reshape(2, 2, 2) / 10)"""),
("""            globals()["load_models"] = lambda seed, pair_run, dev: ("B", lambda x, gt=None: (None, None, o))
            assert "attached ends" in (refusal(live_records, 0, "val_clean", 0.1, 0, device="cpu", pair_run="e324_s_s0", ends="att") or "") and len(seen) == n_seen, "att without attached ends in the output: refused, no fall back"
            assert "attached ends" in (refusal(live_records, 0, "val_clean", 0.1, 0, device="cpu", ends="att") or ""), "att on E312's pair run: refused before the model loads\"""", """            globals()["load_models"] = lambda seed, pair_run, dev: ("B", lambda x, gt=None: (None, None, o))
            assert "attached ends" in (refusal(live_records, 0, "val_clean", 0.1, 0, device="cpu", pair_run="e324_s_s0", ends="att") or "") and len(seen) == n_seen, "att without attached ends in the output: refused, no fall back"
            loaded = []
            globals()["load_models"] = lambda seed, pair_run, dev: loaded.append(pair_run) or ("B", lambda x, gt=None: (None, None, oa))             # a model WITH attached ends: only the check of the pair run can refuse here
            assert "attached ends" in (refusal(live_records, 0, "val_clean", 0.1, 0, device="cpu", ends="att") or "") and not loaded and len(seen) == n_seen, "att on E312's pair run: refused before the model loads\""""),
])
print("ok")
