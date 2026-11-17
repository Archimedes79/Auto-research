import os, sys, torch
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/training/grid")
import jwire_head as JH
os.makedirs(R + "/runs/e325_rulelike", exist_ok=True)
torch.save(dict(state=JH.Wire.rule_like().state_dict(), args=dict(note="hand-set weights: logit = -relu(distance feature): the nearest-piece rule in the shape of the head (fixture of the read-out path)")), R + "/runs/e325_rulelike/model.pt")
print("ok")
