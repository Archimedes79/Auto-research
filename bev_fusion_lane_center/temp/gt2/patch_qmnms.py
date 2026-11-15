p = "training/grid/qmnms.py"
s = open(p, encoding="utf-8").read()
s = s.replace("Differentiable in e (through e_i and the compensation) and in the positions (through the overlaps); returns the logit of e'.",
              "Differentiable in the node's own e and in the positions (through the overlaps and the compensation). NOT in the probability of the stronger neighbour: the decay depends on the overlaps only, the probabilities decide the\norder (a comparison without gradient), so unlike E277's product form the loss cannot raise the stronger node through the weaker one's decay. Returns the logit of e'.")
s = s.replace("    # a gradient reaches the stronger node's probability and both positions", "    # a gradient reaches the weaker node's own probability and both positions (not the stronger node's probability: the order is a comparison)")
s = s.replace("    assert e_logit.grad[0, 1, 1].abs() > 0 and pos.grad[0, :, 1, 2].abs().sum() > 0 and pos.grad[0, :, 1, 1].abs().sum() > 0, (e_logit.grad[0, 1, 1], pos.grad[0, :, 1, 2], pos.grad[0, :, 1, 1])",
              "    assert e_logit.grad[0, 1, 2].abs() > 0 and e_logit.grad[0, 1, 1].abs() == 0 and pos.grad[0, :, 1, 2].abs().sum() > 0 and pos.grad[0, :, 1, 1].abs().sum() > 0, (e_logit.grad[0, 1, 2], e_logit.grad[0, 1, 1], pos.grad[0, :, 1, 2], pos.grad[0, :, 1, 1])")
open(p, "w", encoding="utf-8").write(s)
print("ok")
