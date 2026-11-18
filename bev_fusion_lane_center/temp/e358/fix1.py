import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_data.py", [
("""    import io
    import sys
    import types
    from types import SimpleNamespace as NS
    import jwiring as JWI
    import torch
    up = np.array([0.0, 1.0])
    # 1. names and meta""", """    import io
    import types
    from types import SimpleNamespace as NS
    import jwiring as JWI
    import torch
    up = np.array([0.0, 1.0])
    # 1. names and meta"""),
("""    fakes = dict(lane_gt_data=NS(QData=NS(), loader=lambda *a, **k: (NS(missing=0, __len__=None), [])), qgraph_eval=NS(readout=lambda pred, a, b: ("graph", "bk")),
                 qgraph_readout=NS(parse=lambda s: s, apply=lambda d, spec: "pred"), qpredict=NS(predict=lambda mb, b: [dict(nodes=dict(pos=0, dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0))] * 2))

    class DS:
        missing = 0

        def __len__(self):
            return 2
    batch = dict(x=torch.zeros(2, 1), ent=torch.zeros(2, 1, 4), ent_m=torch.zeros(2, 1, dtype=torch.bool), ext=torch.zeros(2, 1, 4), ext_m=torch.zeros(2, 1, dtype=torch.bool), row=torch.tensor([5, 6]))
    fakes["lane_gt_data"] = NS(QData=NS(), loader=lambda *a, **k: (DS(), [batch]))
""", """
    class DS:
        missing = 0

        def __len__(self):
            return 2
    batch = dict(x=torch.zeros(2, 1), ent=torch.zeros(2, 1, 4), ent_m=torch.zeros(2, 1, dtype=torch.bool), ext=torch.zeros(2, 1, 4), ext_m=torch.zeros(2, 1, dtype=torch.bool), row=torch.tensor([5, 6]))
    fakes = dict(lane_gt_data=NS(QData=NS(), loader=lambda *a, **k: (DS(), [batch])), qgraph_eval=NS(readout=lambda pred, a, b: ("graph", "bk")), qgraph_readout=NS(parse=lambda s: s, apply=lambda d, spec: "pred"),
                 qpredict=NS(predict=lambda mb, b: [dict(nodes=dict(pos=0, dirv=0, e=0, flags=0, section=0), links=dict(src=0, dst=0, p=0, long=0, a=0, b=0))] * 2))
"""),
("zip(ik, (np.hypot(0.2, 1.0), 1.0))", "zip(ik, (np.hypot(0.2, 1.0), np.hypot(0.1, 1.0)))"),
("""                recs_ = live_records(0, "val_clean", 0.1, 0, device="cpu", pair_run="e324_s_s0", ends=ends)
""", """                with contextlib.redirect_stdout(io.StringIO()):
                    recs_ = live_records(0, "val_clean", 0.1, 0, device="cpu", pair_run="e324_s_s0", ends=ends)
"""),
])
print("ok")
