p = "temp/_session_scratch_s292/new/test_mnms_layer.py"
s = open(p, encoding="utf-8").read()
def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
sub('refused(lambda: QMN.matrix_nms_logit(o_p["e"][0], o_p["pos"][0:1].squeeze(0)[None] if False else o_p["pos"], 2.5, 0.1), "a radius above 2 m in the layer")',
    'refused(lambda: QMN.matrix_nms_logit(o_p["e"], o_p["pos"], 2.5, 0.1), "a radius above 2 m in the layer")')
sub('    model.zero_grad()\n    loss, _ = QG.qgraph_loss(model, b, phase)', '    model.zero_grad()\n    torch.manual_seed(7)                                                                      # the loss has a random part (sampling): the same draw for every call\n    loss, _ = QG.qgraph_loss(model, b, phase)')
sub('        n_pairs += int(((ref > 0) & sel & (ref < e)).sum())', '        n_pairs += int((sel & (mine < e - 1e-6)).sum())')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
