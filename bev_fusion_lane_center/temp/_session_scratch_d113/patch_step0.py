import io
p = "temp/_session_scratch_d113/step0_past.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace('''cnt = dict(tw_s=0''', '''wk = []
cnt = dict(tw_s=0''', 1)
s = s.replace('''        k = "w" if name == "w" else "d"''', '''        k = "w" if name == "w" else "d"
        if name == "w":
            wk.append(np.stack([ep[sel], tr.astype(float), sup.astype(float)], 1))''', 1)
s += '''
W = np.concatenate(wk)
print("weak tokens by existence bin: p range | tokens | true share | supported (true) | supported (false)")
for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)):
    m = (W[:, 0] >= lo) & (W[:, 0] < hi)
    t_ = m & (W[:, 1] > 0.5)
    f_ = m & (W[:, 1] < 0.5)
    print("  %.2f - %.2f | %7d | %.3f | %.3f | %.3f" % (lo, hi, m.sum(), t_.sum() / max(m.sum(), 1), W[t_, 2].mean(), W[f_, 2].mean()))
'''
io.open(p, "w", encoding="utf-8").write(s)
