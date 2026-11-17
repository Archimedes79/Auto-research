import os, sys, torch, time
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jends_eval as JEV, jmodel_cost as JC
torch.set_num_threads(2)
m, a = JEV.load_model("e312_p_s0", "cpu")
t = time.time()
c = JC.cost(m)
print({k: (round(v["params_m"], 4), round(v["macs_g"], 4)) for k, v in c.items()}, time.time() - t)
# MHA alone in eval/no_grad vs train
from torch.utils.flop_counter import FlopCounterMode
mha = m.je.blocks[0].att
h = torch.randn(1, 64, 64)
for mode in ("eval", "train"):
    getattr(mha, mode)()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        mha(h, h, h, key_padding_mask=torch.zeros(1, 64, dtype=torch.bool), need_weights=False)
    print(mode, fc.get_total_flops() / 2, "macs; expected", 64*64*192 + 64*64*64 + 2*4*64*64*16)
