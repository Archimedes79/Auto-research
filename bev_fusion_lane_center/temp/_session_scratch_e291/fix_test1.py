p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
s = s.replace("ext = torch.tensor([[[-1.5, -9.5, 0.0, -1.0], [5.5, -9.5, 0.0, -1.0]]] * B)", "ext = torch.tensor([[[-5.5, -9.5, 0.0, -1.0], [5.5, -9.5, 0.0, -1.0]]] * B)")
s = s.replace("import jprior as JP  # noqa: E402\n", "import jprior as JP  # noqa: E402\nimport qjunction as QJ  # noqa: E402\n", 1)
s = s.replace("""def QJ_arms(ds, tk):
    return LDQJ().JG.junction_gt(ds.lg.load(tk))


def LDQJ():
    import qjunction
    return qjunction


""", "")
s = s.replace("QJ_arms(ds, tk)", "QJ.JG.junction_gt(ds.lg.load(tk))").replace("QJ_arms(ds, partner)", "QJ.JG.junction_gt(ds.lg.load(partner))")
s = s.replace('jg = JP.QJ.JG.junction_gt(QV.load_gt(t)) if hasattr(JP, "QJ") else LDQJ().JG.junction_gt(QV.load_gt(t))', 'jg = QJ.JG.junction_gt(QV.load_gt(t))')
s = s.replace("""    assert all(g > 0 for g in gnorm), f"the feature layer has a gradient from step 1: {gnorm}\"""", """    assert all(g > 0 for g in gnorm), f"the feature layer has a gradient from step 1: {gnorm}"
    gate = [float(l.split("gate ")[1].split(" ")[0]) for l in log]
    assert gate[0] == 0.0 and gate[2] > 0.0, f"the gate has no gradient at step 1 (f starts at zero) and has one from step 2 or 3: {gate}\"""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("QJ_arms"), s.count("LDQJ"), "gate = [float" in s)
