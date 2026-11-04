p = "evaluation/bev_modules.py"
s = open(p, encoding="utf-8").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
rep('''    kept = torch.ones(2)
    out_of_place = Recorder(Toy())
    out_of_place.close()
''', '')
rep('''    batch = {"prob": torch.ones(1, 2), "ctx": torch.ones(1, 2), "radar_hits": torch.ones(1, 3), "v": torch.ones(1),''',
    '''    batch = {"prob": torch.ones(1, 2), "ctx": torch.ones(1, 2), "lane_img": torch.ones(1, 3, dtype=torch.uint8), "radar_hits": torch.ones(1, 3), "v": torch.ones(1),''')
rep('''assert c["prob"].abs().sum() == 0 and c["ctx"].abs().sum() == 0 and c["radar"] is batch["radar"]''',
    '''assert c["prob"].abs().sum() == 0 and c["ctx"].abs().sum() == 0 and c["lane_img"].sum() == 0 and c["radar"] is batch["radar"]''')
rep('''and r["prob"].sum() == 2
''', '''and r["prob"].sum() == 2 and r["lane_img"].sum() == 3
''')
open(p, "w", encoding="utf-8").write(s)
