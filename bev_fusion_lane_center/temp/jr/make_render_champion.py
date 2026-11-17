"""Scratch (run once): derive temp/qual_report_1117/render_champion_q1117.py from temp/qual_report_1115/render_q1115.py:
panels = plain B | the champion (token head S1, E366 decode) | E324 s (previous system, rule steps, snapped read-out); the numbers table from the per-frame tables."""
import os
R = os.environ["BFLC_ROOT"]
src = open(os.path.join(R, "temp", "qual_report_1115", "render_q1115.py"), encoding="utf-8").read()


def rep(t, old, new):
    assert t.count(old) == 1, old[:80]
    return t.replace(old, new)


t = src
t = rep(t, '"""scratch (2026-11-15, qualitative report)', '"""scratch (2026-11-17, derived from render_q1115.py for the report of the token champion)')
t = rep(t, 'HERE = os.path.join(ROOT, "temp", "qual_report_1115")\nP1 = pickle.load(open(os.path.join(HERE, "pass1.pkl"), "rb"))',
        'HERE = os.path.join(ROOT, "temp", "qual_report_1115")\nH17 = os.path.join(ROOT, "temp", "qual_report_1117")\nP1 = pickle.load(open(os.path.join(H17, "pass1_with_champion.pkl"), "rb"))')
t = rep(t, 'TAB = {r["k"]: r for r in json.load(open(os.path.join(HERE, "perframe_table.json")))}',
        'TAB = {r["k"]: r for r in json.load(open(os.path.join(HERE, "perframe_table.json")))}\nCH = json.load(open(os.path.join(H17, "perframe_table.json")))["sys"]')
t = rep(t, 'NAMES = (("B", "B: the plain graph"), ("E312", "E312 (junction layer)"), ("E324s", "E324 s (system of record)"))',
        'NAMES = (("B", "plain lane head B"), ("champion_S1_s0", "CHAMPION: token head S1 (E366 decode)"), ("E324s", "E324 s (previous system, rule steps, snapped)"))')
# panel titles: the champion has no own-lane number
a = t.index("    rows = [(\"\", \"B\", \"E312\", \"E324 s\")]")
b = t.index("    tb = axt.table(")
t = t[:a] + '''    kk = k
    ch = CH["champion_S1_s0"]
    rows = [("", "plain B", "champion", "E324 s")]
    rows.append(("topo F1", fmt(t["B_topo"]), fmt(ch["topo_F1"][kk]), fmt(t["E324s_topo"])))
    rows.append(("topo P / R", "-", f"{fmt(ch['topo_P'][kk])} / {fmt(ch['topo_R'][kk])}", "-"))
    rows.append(("own lane R", fmt(t["B_ownR"]), "-", fmt(t["E324s_ownR"])))
    rows.append(("maneuvers", *[f"{rec['sys'][s]['man_ok']}/{rec['sys'][s]['man_n']}" for s in ("B", "champion_S1_s0", "E324s")]))
''' + t[b:]
t = rep(t, 'tb = axt.table(cellText=rows, cellLoc="center", colWidths=[0.46, 0.17, 0.19, 0.19]', 'tb = axt.table(cellText=rows, cellLoc="center", colWidths=[0.34, 0.20, 0.26, 0.20]')
t = rep(t, '''        panel(a, rec, key, f"{name}\\ntopo F1 {fmt(t[key + '_topo'])}   own lane R {fmt(t[key + '_ownR'])}   maneuvers {rec['sys'][key]['man_ok']}/{rec['sys'][key]['man_n']}")''',
        '''        tp = ch["topo_F1"][kk] if key == "champion_S1_s0" else t[key + "_topo"]
        panel(a, rec, key, f"{name}\\ntopo F1 {fmt(tp)}   maneuvers {rec['sys'][key]['man_ok']}/{rec['sys'][key]['man_n']}")''')
t = rep(t, "(Seed 0, VAL.", "(Seed 0, VAL.") if False else t
open(os.path.join(R, "temp", "qual_report_1117", "render_champion_q1117.py"), "w", encoding="utf-8").write(t)
print("ok")
