"""one-off: recursive speed refinement mode for tests/test_flow_explains.py (--recursive, nuScenes)"""
p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1 data: optional GT velocity of F1 (sixth field), own cache name
rep('''def nuscenes_item(path):''', '''def nuscenes_item(path, with_v1=False):''')
rep('''    return maps, torch.from_numpy(vel), torch.from_numpy(g["prev"]["instance"].astype(np.int64)), s01, s12''',
    '''    item = (maps, torch.from_numpy(vel), torch.from_numpy(g["prev"]["instance"].astype(np.int64)), s01, s12)
    if with_v1:                                          # + the GT velocity of F1 (the recursive test scores at F1)
        item = item + (torch.from_numpy(g["cur"]["velocity"].astype(np.float32)),)
    return item''')
rep('''def nuscenes_items(test, n):''', '''def nuscenes_items(test, n, with_v1=False):''')
rep('''    path = f"temp/checks/flow_explains_nusc_{'test' if test else 'train'}_{n}.pt"''',
    '''    path = f"temp/checks/flow_explains_nusc_{'test' if test else 'train'}_{n}{'_v1' if with_v1 else ''}.pt"''')
rep('''        it = nuscenes_item(f)''', '''        it = nuscenes_item(f, with_v1)''')
# 2 model: an optional start field (the carried estimate of the previous pair)
rep('''    def forward(self, f0, f1, steps, v_override=None, vgt=None):''',
    '''    def forward(self, f0, f1, steps, v_override=None, vgt=None, v_init=None):''')
rep('''        v0, corr = r["v_mean"], None''', '''        v0, corr = (r["v_mean"] if v_init is None else v_init), None      # v_init: recursive start (carried estimate)''')
# 3 unpacking tolerates the sixth field
rep("    for maps, vgt, bid, steps, steps12 in test:\n", "    for maps, vgt, bid, steps, steps12, *_ in test:\n")
rep("        maps, vgt, bid, steps, steps12 = test[r]\n", "        maps, vgt, bid, steps, steps12, *_ = test[r]\n")
# 4 the recursive evaluation
rep('''def overlay(a, b):''', '''@torch.no_grad()
def evaluate_recursive(model, test):
    """Recursive speed refinement, scored at F1 against the GT velocity of F1 (vehicles, all / moving > 1 m/s):
    fresh      the field estimated from (F1, F2) alone (hypotheses -> refinement), as in the pair test
    carried    the estimate of (F0, F1) carried to F1 (moved with itself, rotated into F1's axes): constant velocity,
               no new data
    recursive  carried as the START of the (F1, F2) refinement (instead of the fresh hypothesis mean): the prediction
               with the carried estimate is compared with the new frame (correlation, Doppler in every iteration)
    Recursion pays if 'recursive' beats both."""
    acc = {}
    for maps, vgt0, bid, s01, s12, vgt1 in test:
        m = maps[None].to(DEV)
        v01 = model(m[:, 0], m[:, 1], [s01], vgt=vgt0[None].to(DEV))[0][-1]
        carried = carry(v01, [s01])
        fresh = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV))[0][-1]
        recur = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV), v_init=carried)[0][-1]
        veh = maps[1, 1] > 0.5
        masks = {"vehicles": veh, "moving vehicles": veh & (vgt1.norm(dim=0) > 1.0)}
        for name, v in (("fresh", fresh), ("carried", carried), ("recursive", recur)):
            e = (v[0].cpu() - vgt1).norm(dim=0)
            for k, mk in masks.items():
                a = acc.setdefault((name, k), [0.0, 0])
                a[0] += e[mk].sum().item()
                a[1] += int(mk.sum())
    return {f"EPE F1 {k} {name}": s_ / max(1, n_) for (name, k), (s_, n_) in acc.items()}


def overlay(a, b):''')
# 5 main: --recursive loads items with the F1 velocity and prints the recursive scores
rep('''        data, test = nuscenes_items(False, n_train), nuscenes_items(True, n_test)''',
    '''        rec = "--recursive" in sys.argv
        data, test = nuscenes_items(False, n_train, rec), nuscenes_items(True, n_test, rec)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
