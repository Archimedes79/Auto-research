p = "evaluation/bev_errors.py"
s = open(p, encoding="utf-8").read()
old = '''        mine = {"vehicle": [2, 5, 4], "pedestrian": [1, 1, 3], "two_wheeler": [0, 0, 0]}          # tp, n_gt, n_det at min-score 0: vehicles 2 of 5 boxes found with 4 detections incl. the low one'''
new = '''        mine = {"vehicle": [3, 5, 5], "pedestrian": [1, 1, 2], "two_wheeler": [0, 0, 0]}          # tp, n_gt, n_det with every detection counted, counted by hand from the frame above'''
assert old in s
s = s.replace(old, new)
old2 = '''        assert [b[:2] for b in check_against_card(run, 1)] == [("vehicle", "tp"), ("vehicle", "fn"), ("vehicle", "fp")][:0] or check_against_card(run, 1), "a card that differs is reported"'''
new2 = '''        bad = check_against_card(run, 1)
        assert [(b[0], b[1]) for b in bad] == [("vehicle", "tp"), ("vehicle", "fn")] and bad[0][2] == 3 and bad[0][3] == 4, bad              # one more tp on the card: tp and fn (n_gt - n_tp) differ'''
assert old2 in s
s = s.replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
