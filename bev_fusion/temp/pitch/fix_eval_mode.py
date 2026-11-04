p = "evaluation/bev_timing.py"
s = open(p, encoding="utf-8").read()
old = '''    model.train()
    train = train_split(lambda: T.forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=True)[0]["total"], lambda: model.zero_grad(set_to_none=True), dev)
'''
new = '''    model.train()                                                       # (a training-mode forward updates the BatchNorm statistics in memory only; this process ends after the timing)
    try:
        train = train_split(lambda: T.forward_loss(model, loss, batch, 1.0, dev, cfg.amp, cfg, train_terms=True)[0]["total"], lambda: model.zero_grad(set_to_none=True), dev)
    finally:
        model.eval()
'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
