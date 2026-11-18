import io
p = "evaluation/jwire_data.py"
s = io.open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)
rep("o = dict(kp_pos=x[:, :, None].repeat(1, 4, 2), kp_dir=torch.zeros(n, 4, 2), kp_p=torch.zeros(n, 4), valid=torch.ones(n, 2, 2, dtype=torch.bool), logit=torch.zeros(n, 2, 2), allowed=torch.ones(n, 2, 2, dtype=torch.bool))",
    "o = dict(kp_pos=x[:, :, None].repeat(1, 4, 2), kp_dir=torch.zeros(n, 4, 2), kp_p=torch.zeros(n, 4), valid=torch.ones(n, 2, 2, dtype=torch.bool), logit=torch.zeros(n, 2, 2), allowed=torch.ones(n, 2, 2, dtype=torch.bool),\n             att_pos=100.0 + x[:, :, None].repeat(1, 4, 2), att_dir=torch.zeros(n, 4, 2))")
rep('seen.append((pred, float(e["kp_pos"][0, 0]), float(g["ent"][0, 0]), row, ends))', 'seen.append((pred, float(e["kp_pos"][0, 0]), float(g["ent"][0, 0]), row, ends, float(e["att_pos"][0, 0]) if "att_pos" in e else None))')
rep('assert seen == [(float(r), float(r), float(r), r, "att") for r in rows[:5]], seen                      # B\'s prediction, the pair model\'s output and the GT ends of every record belong to the SAME row',
    'assert seen == [(float(r), float(r), float(r), r, "att", 100.0 + r) for r in rows[:5]], seen             # B\'s prediction, the pair model\'s output (keypoints AND attached ends) and the GT ends of every record belong to the SAME row')
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
