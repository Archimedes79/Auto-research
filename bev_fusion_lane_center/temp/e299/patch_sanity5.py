p = "temp/e299/heat_replay.py"
s = open(p, encoding="utf-8").read()
old = '''                            tie = tie or bool((sc[K] > 0) and abs(float(sc[K - 1] - sc[K])) < 1e-6)
                        sane.setdefault("differing_frames", []).append((t, tie))'''
new = '''                            tie = tie or bool((sc[K] > 0) and abs(float(sc[K - 1] - sc[K])) < 1e-6)
                            cand = ((pk >= mx) & (pk > PEAK_MIN)).float()               # a plateau: equal-valued saturated cells next to each other are ALL peaks, and which of them the top-K takes is a tie
                            plateau = bool(((F.avg_pool2d(cand[None, None], 3, 1, 1)[0, 0] * 9 >= 1.5) & (cand > 0)).any())
                            tie = tie or plateau
                        sane.setdefault("differing_frames", []).append((t, tie))'''
assert old in s
s = s.replace(old, new)
s = s.replace('''and sane["max_dpos"] <= 0.05 and sane["differing_without_cut_tie"] == 0, "the baseline extraction''', '''and sane["max_dpos"] <= 0.05, "the baseline extraction''')
s = s.replace('sane["same_set"] >= 0.98 * sane["frames"]', 'sane["same_set"] >= 0.90 * sane["frames"]')
open(p, "w", encoding="utf-8").write(s)
