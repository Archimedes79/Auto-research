import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
("""            JD.save_records(cache("val_records_v1_s0_pair_e324_s_s0.pkl"), [JD.toy_record(i, i % 22) for i in range(60)], 0, "e324_s_s0", "att") if False else None   # (an att meta under the kp name is not constructible: `save_records` names no path)
            kpv = cache("val_records_v1_s0_pair_e324_s_s0.pkl")
            pickle.dump(""", """            kpv = cache("val_records_v1_s0_pair_e324_s_s0.pkl")
            pickle.dump("""),
("""and sorted(os.listdir(rd)) >= ["log.txt", "model.pt", "model_wn.pt", "wire_agreement.json"], os.listdir(rd)""", """and {"log.txt", "model.pt", "model_wn.pt", "wire_agreement.json"} <= set(os.listdir(rd)), os.listdir(rd)"""),
])
print("ok")
