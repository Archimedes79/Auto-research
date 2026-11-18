p = "training/grid/jprior.py"
s = open(p, encoding="utf-8").read()
old = s[s.index("    same = (arm[:, :, None] == arm[:, None, :])"):s.index("    known = has & (pick(n) != UNK)")]
new = '''    own = arm[:, None, :].expand(B, N, N)                                                              # [b, k, i] = the arm of keypoint i
    mem = elig.gather(2, own) & ok[:, :, None] & (kind[:, :, None] == kind[:, None, :])               # [b, k, i]: k is a REAL keypoint of the arm of i, in the direction of i (an arm's keypoints are the ones eligible for it: a keypoint near two arms counts for both)
    lat_k = lat.gather(2, own)                                                                          # the lateral offset of k to the axis of the arm of i
    idx = torch.arange(N, device=pos.device)
    left = mem & ((lat_k > lat_i[:, None, :]) | ((lat_k == lat_i[:, None, :]) & (idx[None, :, None] < idx[None, None, :])))
    rank, found = left.sum(1).float(), mem.sum(1).float()
'''
s = s.replace(old, new)
s = s.replace("""What a keypoint sees: its arm is the one within RADIUS (10 m) of its centre whose travel direction (the heading into the junction for an entry, the opposite for an exit) is within 60 degrees of the keypoint's heading, the nearest to
its axis; the count is the one of its OWN direction.""", """What a keypoint sees: the arms it is eligible for are those within RADIUS (10 m) of their centre whose travel direction (the heading into the junction for an entry, the opposite for an exit) is within 60 degrees of the keypoint's heading; its own arm is
the one of them nearest to its axis, the count is the one of its OWN direction. The keypoints of an arm are all the real ones eligible for it (the quantity of the step 0 of the card).""")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("mem = elig.gather" in s)
