p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
marker = "def snap_table():"
funcs = '''def maneuver_block():
    """E310 / E311: maneuver coverage per variant (runs/<run>/maneuver_val.json of evaluation/jmaneuver.py) as a table and a bar chart"""
    rows, bars = [], []
    for run, label in (("e308_t_s0", "E308 seed 0"), ("e305_e_s0", "E305 seed 0")):
        d = J(f"runs/{run}/maneuver_val.json")
        if not d:
            continue
        for v, r in d["variants"].items():
            rows.append([label, v, f"{r['realised']} / {r['maneuvers']}", f3(r["recall"]), f3(r["junction_precision"]), str(r["frames_none"])])
            bars.append((label + " " + v, r["recall"]))
    if not rows:
        return "<p>pending</p>"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_, ax = plt.subplots(figsize=(11, 4.2))
    cols = ["#7f7f7f" if "plain" in b else ("#ff7f0e" if "oracle" in b else ("#006400" if b.endswith("s") else "#2ca02c")) for b, _ in bars]
    ax.bar(range(len(bars)), [v for _, v in bars], color=cols)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b for b, _ in bars], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("share of GT maneuvers realised")
    ax.set_title("maneuver coverage of the delivered graph (grey plain, green connectors, dark green snapped ends, orange perfect pairs)", fontsize=9)
    fig_.tight_layout()
    pth = os.path.join(HERE, "media", "maneuver_bars.png")
    fig_.savefig(pth, dpi=80)
    plt.close(fig_)
    return tbl(["run", "delivered graph", "maneuvers realised", "share", "junction precision", "frames with none"], rows, num=range(2, 6)) + fig(jpg(pth, "maneuver_bars.jpg"), "Maneuver coverage per variant (evaluation/jmaneuver.py).")


'''
assert marker in s
s = s.replace(marker, funcs + marker, 1)
s = s.replace("{FAIL}<h3>After: E305", "{FAIL}{maneuver_block()}<h3>After: E305", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
