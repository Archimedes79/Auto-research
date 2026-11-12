def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:70])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))


sub("training/grid/jmaps.py", "def jmaps_loss(out, batch):", "def jmaps_loss(out, batch, far_w=0.0, far_r=35.0):")
sub("training/grid/jmaps.py", "    l_h = F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw) + _dice(j1[:, 2:4], jh)\n",
    "    if far_w > 0:                                                                                      # E304: the cells beyond far_r metres from the ego count 1 + far_w times in the heat BCE (53 % of the GT ends are there, the recall is 0.55)\n"
    "        cc = -50.0 + (torch.arange(100, device=dev).float() + 0.5)\n"
    "        wmap = 1.0 + far_w * ((cc[:, None] ** 2 + cc[None, :] ** 2).sqrt() > far_r).float()\n"
    "        l_h = (F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw, reduction=\"none\") * wmap).mean() + _dice(j1[:, 2:4], jh)\n"
    "    else:\n"
    "        l_h = F.binary_cross_entropy_with_logits(j1[:, 2:4], jh, pos_weight=pw) + _dice(j1[:, 2:4], jh)\n")
sub("training/grid/jmaps.py", "        loss, info = jmaps_loss(out, batch)\n        loss.backward()\n        dead =", "        loss, info = jmaps_loss(out, batch)\n        lf, _ = jmaps_loss(out, batch, far_w=2.0)                                                         # E304: the far weight changes the loss and nothing else\n        assert abs(float(lf) - float(loss)) > 1e-6\n        loss.backward()\n        dead =")
sub("training/train/jmaps_train.py", "    ap.add_argument(\"--joint\", action=\"store_true\")\n", "    ap.add_argument(\"--joint\", action=\"store_true\")\n    ap.add_argument(\"--far-w\", type=float, default=0.0, help=\"E304: the heat BCE of the cells beyond 35 m counts 1 + w times\")\n")
sub("training/train/jmaps_train.py", "        loss, info = JM.jmaps_loss(out, b)\n", "        loss, info = JM.jmaps_loss(out, b, far_w=a.far_w)\n")
print("ok")
