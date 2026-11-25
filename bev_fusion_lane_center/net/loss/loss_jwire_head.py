"""Losses of the wiring head: the loss side of net/model/jwire_head.py (D152; the functions are unchanged, `jwire_head` holds the network)."""
import torch
import torch.nn.functional as F



def wire_loss(logits, label):
    """cross entropy over the places of the LABELLED ends only (label >= 0); 0 when no end is labelled (no gradient)"""
    sel = label >= 0
    if not bool(sel.any()):
        return torch.nan_to_num(logits, neginf=0.0).sum() * 0.0
    return F.cross_entropy(logits[sel], label[sel])
