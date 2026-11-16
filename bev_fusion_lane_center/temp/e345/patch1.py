import re
R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/"
p = R + "jprune_probe.py"
s = open(p, encoding="utf-8").read()
old = '''def max_diff(a, b):
    """largest absolute difference over the tensors of two outputs"""
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a if torch.is_tensor(a[k]) and a[k].numel())
'''
new = '''def max_diff(a, b, ints=True):
    """largest absolute difference over the floating-point tensors of two outputs; the integer / bool tensors (keypoint cells, validity) must be equal, else inf (`ints=False`: only the floating-point ones count)"""
    d = 0.0
    for k in a:
        if torch.is_tensor(a[k]) and a[k].numel():
            if a[k].is_floating_point():
                d = max(d, float((a[k] - b[k]).abs().max()))
            elif ints and not torch.equal(a[k], b[k]):
                d = float("inf")
    return d
'''
assert old in s
s = s.replace(old, new)
s = s.replace("d_zero, d_effect, d_orig = 0.0, max(max_diff(a, b) for a, b in zip(base, zout)), 0.0", "d_zero, d_effect, d_orig = 0.0, max(max_diff(a, b, False) for a, b in zip(base, zout)), 0.0")
s = s.replace("d_orig = max(d_orig, max(max_diff(a, b) for a, b in zip(base, run(ablate(model, edits_for(u, 1.0, calib), taps)[0]))))", "d_orig = max(d_orig, max(max_diff(a, b, False) for a, b in zip(base, run(ablate(model, edits_for(u, 1.0, calib), taps)[0]))))")
open(p, "w", encoding="utf-8").write(s)
p = R + "jprune_table.py"
s = open(p, encoding="utf-8").read()
old = 'keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "id", "unit", "f"), k))'
new = 'first = ["model", "id", "unit", "f", "group", "saved_gmacs", "saved_params", "klass", "label", "class_loss"]\n    keys = first + sorted({k for r in rows for k in r} - set(first))'
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
p = R + "tests/test_jprune_probe.py"
s = open(p, encoding="utf-8").read()
old = "max(PP.max_diff(a, b) for a, b in zip(got, base)) > 1e-6, name"
new = "max(PP.max_diff(a, b, False) for a, b in zip(got, base)) > 1e-6, name"
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
