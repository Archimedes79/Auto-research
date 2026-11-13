import re
p = "temp/e309/lines_stats.py"
s = open(p, encoding="utf-8").read()
s = s.replace('print(f"seed {seed}, {len(PF)} frames', 'RES = {}\nprint(f"seed {seed}, {len(PF)} frames')
s = s.replace('    print(f"{k:8s} {np.mean(r[\'n_lines\']):11.1f}', '    RES[k] = dict(lines_per_frame=float(np.mean(r["n_lines"])), length_m=float(np.mean(r["length"])), bad1=bad1, bad2=bad2, turn10=float(100 * (T > 10).sum() * DS / max(totlen, 1e-9)), turn25=float(100 * (T > 25).sum() * DS / max(totlen, 1e-9)), joins_per_frame=len(r["joins"]) / len(PF), join_med=float(np.median(j)), join_p90=float(np.percentile(j, 90)))\n    print(f"{k:8s} {np.mean(r[\'n_lines\']):11.1f}')
s += '\nRES["e312_extra_over_B"] = dict(length_m=extra["len"] / extra["n"], bad1=100 * extra["bad1"] / max(extra["len"], 1e-9), bad2=100 * extra["bad2"] / max(extra["len"], 1e-9))\nimport json\njson.dump(RES, open(os.path.join(ROOT, "temp", "e309", f"lines_stats_s{seed}.json"), "w"), indent=1)\n'
open(p, "w", encoding="utf-8").write(s)
p = "temp/e309/offsets.py"
s = open(p, encoding="utf-8").read()
s = s.replace("out = {}\n", "out = {}\nRES = {}\n")
s = s.replace('    for lo, hi in ((0, 15), (15, 30), (30, 48)):', '    RES[k] = dict(n=len(mo), off_med=float(np.median(mo)), off_p90=float(np.percentile(mo, 90)), jit_med=float(np.median(jt)), jit_p90=float(np.percentile(jt, 90)), off_share=float(100 * tot_m2 / (tot_m2 + tot_j2)), by_range={})\n    for lo, hi in ((0, 15), (15, 30), (30, 48)):\n        RES[k]["by_range"][f"{lo}-{hi}"] = dict(n=int(((rg >= lo) & (rg < hi)).sum()), off_med=float(np.median(mo[(rg >= lo) & (rg < hi)])), jit_med=float(np.median(jt[(rg >= lo) & (rg < hi)])))')
s += '\nimport json\njson.dump(RES, open(os.path.join(ROOT, "temp", "e309", f"offsets_s{seed}.json"), "w"), indent=1)\n'
open(p, "w", encoding="utf-8").write(s)
p = "temp/e309/scenes.py"
s = open(p, encoding="utf-8").read()
s += '\nimport json\njson.dump([dict(scene=str(a), n=int(b), B=float(c), e312=float(d), first_row=int(f)) for a, b, c, d, e, f in sorted(rows, key=lambda t: int(t[0]))], open(os.path.join(ROOT, "temp", "e309", "scenes_s0.json"), "w"), indent=1)\n'
open(p, "w", encoding="utf-8").write(s)
print("patched")
