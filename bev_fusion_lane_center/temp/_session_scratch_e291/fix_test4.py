p = "temp/e291/test_e291_staged.py"
s = open(p, encoding="utf-8").read()
a = s.index("    ck = maps_ckpt(1.5, \"maps_dump.pt\")")
b = s.index("    ns = argparse.Namespace(run=\"p_s0\"")
new = '''    e312 = torch.load(os.path.join(ROOT, "runs", "e312_p_s0", "model.pt"), map_location="cpu")                       # the stored E312 model: real keypoints (a random-weight head has no peak above 0.1)
    for name, prior in (("p_s0", True), ("c_s0", False)):
        m = JT.JEndModel(e312["args"]["maps_args"], order=True, evidence=True, scales=True, e2e=True, prior=prior)
        m.load_state_dict(e312["state"], strict=not prior)
        torch.manual_seed(0)
        if prior:                                                                               # open the gate: a trained term
            for p in m.je.prior.parameters():
                torch.nn.init.normal_(p, 0.0, 0.5)
        os.makedirs(os.path.join(root, "runs", name), exist_ok=True)
        torch.save(dict(state=m.state_dict(), args=dict(e312["args"], lane_prior=prior)), os.path.join(root, "runs", name, "model.pt"))
        open(os.path.join(root, "runs", name, "config.json"), "w").write("{}")
'''
s = s[:a] + new + s[b:]
s = s.replace('max=8, prior_mode="clean,noise,pm1,u,s", prior_noise=0.3)', 'max=16, prior_mode="clean,noise,pm1,u,s", prior_noise=0.3)')
s = s.replace("assert len(fr[n]) == 8 and set", "assert len(fr[n]) == 16 and set")
s = s.replace("for i in range(8)", "for i in range(16)")
s = s.replace('thr=0.1, max=8, prior_mode="clean", prior_noise=0.0))', 'thr=0.1, max=16, prior_mode="clean", prior_noise=0.0))')
s = s.replace('JV.dump(argparse.Namespace(run="c_s0", thr=0.1, max=8, prior_mode=None, prior_noise=0.0))', 'JV.dump(argparse.Namespace(run="c_s0", thr=0.1, max=16, prior_mode=None, prior_noise=0.0))')
s = s.replace('thr=0.1, max=8, prior_mode="clean", prior_noise=0.0))\n        raise RuntimeError("a model without', 'thr=0.1, max=16, prior_mode="clean", prior_noise=0.0))\n        raise RuntimeError("a model without')
s = s.replace('''    old = pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))[0]
    new = pickle.load(open(os.path.join(root, "runs", "c_s0", "ends_val.pkl"), "rb"))[0]
    assert set(new) == set(old) and "prior" not in new, "flag off: the dump has the keys of the stored E312 dump"''', '''    old = pickle.load(open(os.path.join(ROOT, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))[:16]
    new = pickle.load(open(os.path.join(root, "runs", "c_s0", "ends_val.pkl"), "rb"))
    assert len(new) == 16 and all(set(n_) == set(o_) and "prior" not in n_ and n_["token"] == o_["token"] for n_, o_ in zip(new, old)), "flag off: the dump has the keys and the frames of the stored E312 dump"
    assert all(np.array_equal(n_["valid"], o_["valid"]) and np.abs(n_["kp_pos"] - o_["kp_pos"]).max() < 1e-2 for n_, o_ in zip(new, old)), "flag off: the keypoints of the stored E312 dump (made on the GPU) are reproduced"''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print(s.count("max=16"), "maps_dump" in s)
