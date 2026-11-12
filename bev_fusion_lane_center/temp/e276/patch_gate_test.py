p = "training/grid/test_soft_nms.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''# 11. a batch is a stack''', '''# the phase gate reads the same damped logit (critic pass 2026-11-08: `_node_stats` used to call the raw node head)
sys.path.insert(0, os.path.join(HERE, "..", "train"))
import qgates as QGt  # noqa: E402

with torch.no_grad():
    _, no_g = QGt._node_stats(stub_model((1.5, 0.002, 2)), {"x": x_}, "cpu")
check(torch.allclose(no_g["e"], e_two["e"]) and not torch.allclose(no_g["e"], lg_c), "the gate must read the damped logit")
# 11. a batch is a stack''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
