p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
old = '''    assert all(np.array_equal(n_["valid"], o_["valid"]) and np.abs(n_["kp_pos"] - o_["kp_pos"]).max() < 1e-2 for n_, o_ in zip(new, old)), "flag off: the keypoints of the stored E312 dump (made on the GPU) are reproduced"'''
new = '''    canon = lambda f: (lambda q: q[np.lexsort((q[:, 1], q[:, 0]))])(f["kp_pos"][f["valid"].reshape(-1)])  # noqa: E731    (the order of keypoints with equal scores differs between the GPU and the CPU: compare the sets)
    assert all(n_["valid"].sum() == o_["valid"].sum() and np.abs(canon(n_) - canon(o_)).max() < 1e-2 for n_, o_ in zip(new, old)), "flag off: the keypoints of the stored E312 dump (made on the GPU) are reproduced"'''
assert old in s
s = s.replace(old, new)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
