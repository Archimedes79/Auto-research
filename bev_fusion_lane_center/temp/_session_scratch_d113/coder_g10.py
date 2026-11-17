p = "training/grid/test_jpast_train.py"
s = open(p, encoding="utf-8").read()
a = '''    (T0, "u is not logged", "            info[\\"u\\"] = float(np.mean(u_hist)) if u_hist else 0.0", "            pass"),
    (T0, "u is the last step's only", "float(np.mean(u_hist)) if u_hist else 0.0", "float(u_hist[-1]) if u_hist else 0.0"),'''
b = '''    (T0, "u is not logged", "            info[\\"u\\"], info[\\"u_none\\"] = (float(np.mean(h)) if h else 0.0 for h in (u_hist, un_hist))", "            pass"),
    (T0, "u is the last step's only", "(float(np.mean(h)) if h else 0.0 for h in (u_hist, un_hist))", "(float(h[-1]) if h else 0.0 for h in (u_hist, un_hist))"),'''
print(s.count(a))
