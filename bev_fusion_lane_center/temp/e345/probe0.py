import os, sys, torch
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jends_eval as JEV
torch.set_num_threads(2)
for run in ("e312_p_s0", "e324_s_s0"):
    m, a = JEV.load_model(run, "cpu")
    print(run, a["maps_args"], "scales", a.get("scales"), "attach", a.get("attach"), "unc", a.get("unc"), "ntok", a.get("ntok"))
    print([n for n, _ in m.named_children()], [n for n, _ in m.jmodel.named_children()], [n for n, _ in m.jmodel.jm.named_children()], [n for n, _ in m.je.named_children()])
    print(sum(p.numel() for p in m.parameters()) / 1e6)
