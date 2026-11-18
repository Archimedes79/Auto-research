set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/grid/test_jpast_train.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''    assert E.donor_plan([-1, 0, 1]) == [None, None, None] or True
''', '''''') if False else None
rep('''    assert torch.equal(trunc[3]["logp"], cold[3]) and torch.equal(trunc[7]["logp"], cold[7]) and not torch.equal(trunc[2]["logp"], cold[2]) and torch.equal(trunc[2]["logp"], stream[2][0]["logp"]) and not torch.equal(trunc[4]["logp"], stream[4][0]["logp"]) is False
''', '''    assert torch.equal(trunc[3]["logp"], cold[3]) and torch.equal(trunc[7]["logp"], cold[7]), f"{label}: truncated is cold at position 3"
    assert all(torch.equal(trunc[i]["logp"], stream[i][0]["logp"]) for i in (1, 2, 5, 6)) and not any(torch.equal(trunc[i]["logp"], stream[i][0]["logp"]) for i in (3, 7, 8)), f"{label}: truncated = stream until the reset, then another state"
''')
rep('''    assert E.donor_plan([-1, 0, 1, -1]) == [None, None, None, None] or E.donor_plan([-1, 0, 1, -1]) == [None, 3 - 3 + 3 if False else None, None, None] or True
    assert E.donor_plan([-1, 0, 1, 2, -1, 4]) == [None, 4, None, None, None, 0], E.donor_plan([-1, 0, 1, 2, -1, 4])
''', '''    assert E.donor_plan([-1, 0, 1, 2, -1, 4]) == [None, 4, 5, None, None, 0], E.donor_plan([-1, 0, 1, 2, -1, 4])      # the donor scene (2 frames) has no step 3
''')
rep('''    assert u1 > 0 and u2 > u1 * 0.5 and u2 < u1 * 1.01 + 1e-9 or u2 > 0, f"{label}: u {u1} {u2}"''', '''    assert u1 > 0 and abs(u2 / u1 - 4.0 / 3.0) < 1e-4, f"{label}: u is the mean of the last steps: {u1} {u2} (steps 1 - 3: 0, ua, ua)"''')
rep('''        za = T.parse_args(["--name", "z", "--smoke", "--bs", "2", "--seq", "3", "--no-past"])''', '''        ia = T.parse_args(["--name", "i", "--smoke", "--bs", "2", "--seq", "3", "--seed", "0", "--init-seed", "3"])
        T.train(ia, runs=tmp, ckpt_every=10, smoke_k=K)
        mi, ai = T.load_model("i", "cpu", tmp)
        assert json.load(open(os.path.join(tmp, "i", "config.json")))["seed"] == 3 and ai["seed"] == 0 and ai["init_seed"] == 3 and json.load(open(os.path.join(tmp, "r", "config.json")))["seed"] == 0, f"{label}: --init-seed is the seed of the loop"
        za = T.parse_args(["--name", "z", "--smoke", "--bs", "2", "--seq", "3", "--no-past"])''')
rep('''and meta["use"]["has_prev"][0] is False or meta["use"]["has_prev"][0] == False  # noqa: E712''', '''and meta["use"]["has_prev"][0] is False''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
