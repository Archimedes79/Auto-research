"""E369 probe (throw-away): the fixture on the same frames with the scale head set to 1/3 at step 0 (sc_o weight 0, bias logit((1/3 - 0.05) / 1.2))"""
import math
import os
import sys
import types

import torch

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as E  # noqa: E402

real = E.JKT.build_head


def third(args):
    m = real(args)
    torch.nn.init.zeros_(m.sc_o.weight)
    m.sc_o.bias.data.fill_(math.log((1 / 3 - 0.05) / 1.2 / (1 - (1 / 3 - 0.05) / 1.2)))
    return m


E.JKT.build_head = third
torch.set_num_threads(2)
E.fixture(types.SimpleNamespace(seed=0, run=None, ident="e292_a", split="train", every=3, max=int(sys.argv[1]), lam=E.JK.LAM, mu=E.JK.MU), out_dir=os.path.join(ROOT, "temp", "e369", "out"))
