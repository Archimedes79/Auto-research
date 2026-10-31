p = "temp/loop21/fig_status_frames.py"
L = open(p, encoding="utf-8").read().split("\n")
def setl(n, text):
    L[n - 1] = text
setl(192, '    axs[0, 0].set_title(f"camera, frame {i}, {LOCN[int(loc[i])]}", fontsize=9)')
setl(194, '    axs[0, 1].set_title("1  BEV features (frozen), mean |value|", fontsize=9)')
setl(196, '    axs[0, 2].set_title("2  dense lane heat", fontsize=9)')
setl(198, '    axs[0, 3].set_title("3  dense direction (hue = heading)", fontsize=9)')
setl(207, '    ax.set_title("4  hypothesis tree, colour = path validity", fontsize=9)')
setl(211, '    axs[1, 1].set_title("5  junction heat: split red, merge blue", fontsize=9)')
setl(218, '        ax2.set_title(ttl + f", F1 {f1(g, gts[i]):.2f}", fontsize=9)')
s = "\n".join(L)
s = s.replace('f"6  assembled graph (path union, anchoring, pruning):\n{nv(G[\'recall\'][i])} vertices (node ids)"', 'f"6  assembled: {nv(G[\'recall\'][i])} vertices"')
s = s.replace('f"7  after the output stage simplify_dp 0.2 m:\n{nv(G[\'recall_dp\'][i])} vertices"', 'f"7  after simplify_dp: {nv(G[\'recall_dp\'][i])} vertices"')
s = s.replace('fig.suptitle("Inside the network: system `recall` on one VAL frame (ego frame, x right, y forward, +-50 m; 1 - 3 inputs, 4 - 5 builder outputs, 6 - 7 assembly)", fontsize=10)', 'fig.suptitle("system recall on one VAL frame (ego frame, +-50 m): 1 - 3 inputs, 4 - 5 builder outputs, 6 - 7 assembly", fontsize=10)')
open(p, "w", encoding="utf-8").write(s)
print("ok")
