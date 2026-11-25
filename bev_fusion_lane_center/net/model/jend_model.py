"""The junction system as one model (E287 and its arms): the junction maps head (`jmaps`), B's frozen node head and the junction pair head (`jends`).
Trained by training/train/jends_train.py; here since D152 (the class is unchanged)."""
import torch
import torch.nn as nn

import jends as JE
import jmaps as JM
import qnodes as QN


class JEndModel(nn.Module):
    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False, e2e=False, ntok=False, unc=False, attach=False):
        super().__init__()
        self.e2e = e2e
        assert not (evidence and "nozone" in maps_args.get("ctx", "dil")), "evidence on a maps head without the zone / core supervision (E332 arm b) would read unsupervised channels: train the pair head with --no-evidence"
        self.jmodel = JM.JModel(width=48, d=maps_args.get("d", 64), ctx=maps_args.get("ctx", "dil"))
        self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=maps_args.get("d", 64), scales=scales, ntok=64 if ntok else 0, unc=unc, attach=attach, paint_ev="nopaint" not in maps_args.get("ctx", "dil"))   # E341: a maps head without the 0.5 m branch gives the evidence without the two paint maxima
        self.nodes = QN.NodeHead(c2=3 * 48, c1=48, c05=32, d=64) if (ntok or attach) else None           # E308: B's frozen node head: its tokens at the keypoints; E320: its nodes are the candidates of the attached connector ends

    def forward(self, x, gt=None):
        if self.e2e:                                                                                     # the maps head (jmodel.jm) gets the gradient of the pair loss
            P, out = self.jmodel(x, joint=False)
        else:
            with torch.no_grad():
                P, out = self.jmodel(x, joint=False)
        tok = nd = None
        if self.nodes is not None:
            with torch.no_grad():
                nd = self.nodes(P["feat2"], P["feat1"], P["feat05"], P["road2"], P["centre1"])
            tok = nd["tok"]
        return P, out, self.je(out, P, gt=gt, tok=tok, nodes=nd if self.je.attach is not None else None)
