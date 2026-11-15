"""one-off: the scoring mechanics of jprune_table.score_point on a point folder that holds the STORED dump of E312 s0 (so the card must equal the stored snapped card)"""
import json, os, shutil, sys, tempfile
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import torch
torch.set_num_threads(2)
import jends_eval as JEV
import qends_eval as QE
import jprune_table as PT
tmp = tempfile.mkdtemp(dir=os.path.join(R, "temp/e345"))
pr = os.path.join(tmp, "runs", PT.point_run("e312_p_s0", 0))
os.makedirs(pr)
os.symlink(os.path.join(R, "runs/e312_p_s0/model.pt"), os.path.join(pr, "model.pt"))
shutil.copy(os.path.join(R, "runs/e312_p_s0/ends_val.pkl"), os.path.join(pr, "ends_val.pkl"))
os.symlink(os.path.join(R, "runs/e273_p_s0"), os.path.join(tmp, "runs/e273_p_s0"))
os.makedirs(os.path.join(tmp, "temp"))
os.symlink(os.path.join(R, "temp/e276"), os.path.join(tmp, "temp/e276"))
JEV.ROOT = QE.ROOT = tmp
PT.score_point("e312_p_s0", "e312", 0, os.path.join(tmp, "runs"), readouts=["snap"])
a = json.load(open(os.path.join(pr, PT.card_stem(0, PT.READOUTS["e312"]["snap"]) + ".json")))["numbers"]["head"]
b = json.load(open(os.path.join(R, "runs/e312_p_s0/ends_card_snap_thr0.1.json")))["numbers"]["head"]
print("files", sorted(os.listdir(pr)))
print("max |diff| of the card numbers:", max(abs(a[k] - b[k]) for k in a if isinstance(a[k], float)), "topo_F1", a["topo_F1"], b["topo_F1"])
