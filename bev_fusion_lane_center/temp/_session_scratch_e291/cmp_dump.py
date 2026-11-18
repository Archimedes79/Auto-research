import pickle, numpy as np
new = pickle.load(open("temp/_session_scratch_e291/dump_root/runs/c_s0/ends_val.pkl", "rb"))
old = pickle.load(open("runs/e312_p_s0/ends_val.pkl", "rb"))[:16]
for n, o in zip(new, old):
    vm = int((n["valid"] != o["valid"]).sum())
    both = n["valid"].reshape(-1) & o["valid"].reshape(-1)
    d = np.abs(n["kp_pos"] - o["kp_pos"])[both].max() if both.any() else -1
    lg = np.abs(n["logit"] - o["logit"])[n["allowed"] & o["allowed"]]
    print(n["token"][:6], "valid mismatches", vm, "n valid", int(o["valid"].sum()), "max kp_pos diff (both valid) %.5f" % d, "max logit diff %.4f" % (lg.max() if lg.size else -1), "allowed equal", bool((n["allowed"] == o["allowed"]).all()))
