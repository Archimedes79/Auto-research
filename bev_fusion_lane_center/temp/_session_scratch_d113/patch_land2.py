import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "landing", "land_patch.py")
s = open(p, encoding="utf-8").read()
anchor = "# ------------------------------------------------------------------------------------------------ run_all_tests.sh:"
assert s.count(anchor) == 1
add = '''# the README mentions inside rows that stay
rep("training/README.md", " `loader-fixture` (the token test again through the TRAINING path: the loader's items and `jpast.make_past`, the pose entry against a fresh `rel_pose` of the pair; the identity pose, the previous item's entry and a shifted previous frame must FAIL),", "")
rep("training/README.md", "; `jrecur_train.py` (E354: the loop of `jends_train.py` over sequences of keyframes with the recursive feature state of `grid/jrecur.py`: `--seq K --bs B --state on|off`, the E341 loss over all B x K frames as one batch, one backward pass through the states, `state_contrib` / gate / clock lines in the log; tests `grid/test_jrecur.py`)", "")
rep("tools/README.md", "(what `maintenance/keep_alive.sh` did)", "(what the old keep_alive.sh did; archive tag archive/jtok2-jrecur-2026-11-18 is not its home: git history)")

'''
s = s.replace(anchor, add + anchor)
open(p, "w", encoding="utf-8").write(s)
print("ok")
