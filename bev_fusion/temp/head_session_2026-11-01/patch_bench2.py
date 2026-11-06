import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_loss2.py")
s = open(p, encoding="utf-8").read()
a = s.index("def only(kind_filter):")
b = s.index("fams = [")
new = '''idx = {}
for i, (_, kind, _) in enumerate(lm.spec):
    idx.setdefault(kind, []).append(i)


def tg(kind):
    return torch.cat([T[lm.spec[i][2]] for i in idx[kind]], 1)


def areas():
    x, t, w = sem[:, idx["area"]], tg("area"), W["area"]
    return (L.soft_bce(x, t, w) + 0.5 * L.soft_iou_loss(torch.sigmoid(x), t, w)).mean()


def areas_bce_only():
    return L.soft_bce(sem[:, idx["area"]], tg("area"), W["area"]).mean()


def lines():
    w = W["line"]
    return L.tolerant_dice_loss(torch.sigmoid(sem[:, idx["line"]]), tg("line"), w, 1).mean() + \
        L.dist_huber(sem[:, idx["dist"]], tg("dist"), w, cfg.huber_delta, cfg.dist_near, cfg.dist_w_far, cfg.dist_max).mean()


def objects():
    x, t, w = sem[:, idx["object"]], tg("object"), W["object"]
    return (L.soft_bce(x, t, w) + L.soft_iou_loss(torch.sigmoid(x), t, w)).mean()


'''
s = s[:a] + new + s[b:]
s = s.replace('("areas   BCE + 1-IoU, 5 ch   (x3 dt)", only("areas")),', '("areas   BCE + 0.5 (1-IoU), 5 ch   (x3 dt)", areas),\n    ("   same, BCE only (no IoU term)", areas_bce_only),')
s = s.replace('("lines   dilated Dice + Huber (x3 dt)", only("lines")),', '("lines   dilated Dice + Huber, 2 ch (x3 dt)", lines),')
s = s.replace('("dense objects BCE + IoU, 3 ch (x3 dt)", only("objects")),', '("dense objects BCE + IoU, 3 ch (x3 dt)", objects),')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
