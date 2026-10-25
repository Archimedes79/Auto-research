const fs=require("fs");
let p="C:/ML-Experiment/bev_fusion_lane_center/evaluation/sparse_lanes/bootstrap.py";let s=fs.readFileSync(p,"utf8");
s=s.replace('    m["S"], m["J"], m["Sp"] = SM.composite(m)\n    return m','    m["S"], m["J"], m["Sp"] = SM.composite(m)\n    m["Q"] = SM.quality(m)\n    return m');
s=s.replace('KEYS = ("S", ','KEYS = ("S", "Q", ');
fs.writeFileSync(p,s);
p="C:/ML-Experiment/bev_fusion_lane_center/evaluation/examples/run_examples.py";s=fs.readFileSync(p,"utf8");
s=s.replace('"Reading: S and F1 of the champion','"Reading: Q = S without the sparsity term (rescaled; `sparse_metrics.quality`). S and F1 of the champion');
fs.writeFileSync(p,s);
