import os

p = os.path.join(os.environ["BFLC_ROOT"], "training/data/lane_map_data.py")
s = open(p, encoding="utf-8").read()

old = '''        deliv = valid & (ep >= 0.5)
        weak = valid & ~deliv
        near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
        miss = ~((near & deliv[:, None]).any(0))
        sel = dict(novel=weak & (near & miss[None]).any(1), dup=weak & near.any(1) & ~(near & miss[None]).any(1), false=weak & ~near.any(1), deliv_true=deliv & near.any(1), deliv_false=deliv & ~near.any(1))
'''
new = '''        weak, near, novel, sel = token_roles(pos, dirv, valid, ep, gpos, gdir)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = "bins.append(np.stack([ep[weak], sup[weak], near.any(1)[weak], (near & miss[None]).any(1)[weak]], 1))"
assert s.count(old2) == 1
s = s.replace(old2, "bins.append(np.stack([ep[weak], sup[weak], near.any(1)[weak], novel[weak]], 1))")

func = '''def token_roles(pos, dirv, valid, ep, gpos, gdir):
    """the roles of the tokens of ONE frame against its GT nodes: pos, dirv, valid, ep (K, ...) the tokens of B' (ep = the existence probability), gpos, gdir (M, 2) the GT nodes -> (weak (K,) = valid and ep < 0.5, near (K, M) = within RAD of the node with a heading cosine >= COS,
    novel (K,) = near a GT node that NO delivered token (valid, ep >= 0.5) is near, roles = dict of (K,) masks: novel (weak), dup (weak, near found nodes only), false (weak, near none), deliv_true / deliv_false (delivered, near a node or not))"""
    deliv = valid & (ep >= 0.5)
    weak = valid & ~deliv
    near = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
    miss = ~((near & deliv[:, None]).any(0))
    nov = (near & miss[None]).any(1)
    return weak, near, nov, dict(novel=weak & nov, dup=weak & near.any(1) & ~nov, false=weak & ~near.any(1), deliv_true=deliv & near.any(1), deliv_false=deliv & ~near.any(1))


'''
anchor = "def potential(split="
assert s.count(anchor) == 1
s = s.replace(anchor, func + anchor)

test_old = '''    # the pose error by hand: p' = R(yaw) p + shift'''
test_new = '''    # the roles of the weak and delivered tokens against the GT nodes, by hand: two nodes 2 m apart heading north; the boundaries are inclusive (a token exactly 1.5 m from a node, a heading cosine of exactly 0.5, a probability of exactly 0.5)
    g_pos, g_dir = np.array([[0.0, 0.0], [0.0, 2.0]]), np.tile([0.0, 1.0], (2, 1))
    t_pos = np.array([[0.0, 0.1], [0.0, 2.2], [0.1, 0.2], [20.0, 20.0], [1.5, 2.0], [0.0, 2.1], [30.0, 30.0], [0.0, 2.0], [-30.0, -30.0]])
    t_dir = np.tile([0.0, 1.0], (9, 1))
    t_dir[5] = [0.8660254037844386, 0.5]
    t_val = np.array([True] * 7 + [False, True])
    t_ep = np.array([0.9, 0.3, 0.2, 0.1, 0.3, 0.4, 0.7, 0.9, 0.5])
    weak, near, nov, roles = token_roles(t_pos, t_dir, t_val, t_ep, g_pos, g_dir)
    want = dict(novel=[1, 4, 5], dup=[2], false=[3], deliv_true=[0], deliv_false=[6, 8])                  # token 0 delivered near node 0; 1, 4, 5 weak near the missed node 1; 2 weak near the found node 0; 7 is padding (no role)
    assert {k: np.flatnonzero(v).tolist() for k, v in roles.items()} == want and weak.tolist() == [False, True, True, True, True, True, False, False, False] and nov.tolist() == [False, True, False, False, True, True, False, True, False], roles
    assert near[4].tolist() == [False, True] and near[5].tolist() == [False, True] and near[2].tolist() == [True, False] and not near[3].any(), "near: inclusive radius and cosine"
    # the pose error by hand: p' = R(yaw) p + shift'''
assert s.count(test_old) == 1
s = s.replace(test_old, test_new)
open(p, "w", encoding="utf-8").write(s)
print("patched")
