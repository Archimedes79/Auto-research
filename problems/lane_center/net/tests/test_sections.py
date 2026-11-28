"""Test of the section stage of the lane head (`net/model/sections.section_membership`) with its loss (`net/loss/loss_sections.section_loss`): two chains get their own section starts, a false link does
not join them, the gradient reaches the link logits and the section-start logits. CPU, seconds."""
import torch

from loss_sections import section_loss
from qnodes import N
from sections import section_membership


def main():
    torch.manual_seed(0)
    B, K = 1, 12
    # two chains: nodes 0..5 (section start 0) and 6..11 (section start 6); true links i -> i+1 inside a chain
    valid = torch.ones(B, K, dtype=torch.bool)
    pb, pa, pc = [], [], []
    for a in list(range(0, 5)) + list(range(6, 11)):
        pb.append(0), pa.append(a), pc.append(a + 1)
    # false candidate links: 5 -> 6 (chains are not joined), 11 -> 0
    pb += [0, 0]
    pa += [5, 11]
    pc += [6, 0]
    logit = torch.tensor([4.0] * 10 + [-4.0, -4.0])
    links = dict(act=torch.arange(K)[None], valid=valid, pb=torch.tensor(pb), pa=torch.tensor(pa), pc=torch.tensor(pc), logit=logit.clone().requires_grad_(True))
    s_logit = torch.full((B, N, N), -4.0)
    s_logit.view(B, -1)[0, 0] = 4.0
    s_logit.view(B, -1)[0, 6] = 4.0
    M, Pb = section_membership(links, s_logit)
    start = M.argmax(-1)[0]
    assert start[:6].eq(0).all() and start[6:].eq(6).all(), start
    assert float(M[0, 5, 0]) > 0.9, M[0, 5, :8]
    # a gradient reaches the link logits and the section-start logits
    gt = dict(piece=torch.full((B, N, N), -1, dtype=torch.long), sstart=torch.zeros(B, N, N))
    gt["piece"].view(B, -1)[0, :6], gt["piece"].view(B, -1)[0, 6:12] = 0, 1
    gt["sstart"].view(B, -1)[0, 0], gt["sstart"].view(B, -1)[0, 6] = 1, 1
    sl = s_logit.clone().requires_grad_(True)
    M2, _ = section_membership(links, sl)
    ce, pl = section_loss(M2, gt, links)
    (ce + pl).backward()
    assert links["logit"].grad is not None and float(links["logit"].grad.abs().sum()) > 0 and float(sl.grad.abs().sum()) > 0
    print("start CE", float(ce), "pair loss", float(pl))
    assert float(ce) < 0.2
    print("test_sections ok")


if __name__ == "__main__":
    main()
