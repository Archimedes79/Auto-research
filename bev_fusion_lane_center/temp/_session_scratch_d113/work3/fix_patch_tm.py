import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp/_session_scratch_d113/work3/patch_tm.py")
s = open(p, encoding="utf-8").read()
old = """        r_ = E.JE.read_dump(argparse.Namespace(run="x", seed=0, split="train", model="model.pt", every=1, drop=0.0, drop_seed=0), tmp, 6)
        assert r_["idx"].tolist() == want_idx and len(r_["fr"]) == 6
"""
new = """        mt = dtr["meta"]                                                                                  # what the scorer's reader checks (`jtoken_eval.read_dump`; it also loads the TRAIN GT of the whole split, not done here)
        assert mt["run"] == "x" and mt["model"] == "model.pt" and mt["every"] == 1 and mt["drop"] == 0.0 and mt["drop_seed"] == 0 and mt["mode"] == "err" and len(dtr["frames"]) == 6
"""
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
