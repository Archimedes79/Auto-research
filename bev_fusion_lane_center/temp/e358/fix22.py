import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
("""            n_unjit = len(real_tensors([JD.toy_record(i, i % 22) for i in range(30)])[2])
""", """            unjit = real_tensors([JD.toy_record(i, i % 22) for i in range(30)])
            n_unjit = len(unjit[2])
            assert wa["no_jitter_head"]["train"]["oracle"] == int((unjit[3] & (unjit[2] >= 0)).sum()) > 0, "wn is judged on the unperturbed TRAIN ends"
"""),
])
print("ok")
