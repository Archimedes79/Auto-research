import json, os
R = os.environ["BFLC_ROOT"]
rows = json.load(open(f"{R}/temp/qual_report_1115/perframe_table.json"))
def show(r):
    print(f"k{r['k']:4d} row{r['row']:5d} sc{r['scene']:2d} {r['token'][:8]} | topo B {r['B_topo']:.2f} E312 {r['E312_topo']:.2f} E324s {r['E324s_topo']:.2f} | ownR B {r['B_ownR']:.2f} E324s {r['E324s_ownR']:.2f} ownP {r['E324s_ownP']:.2f} | far {r['B_far']:.2f}/{r['E324s_far']:.2f} | dup {r['B_dup']:.2f}/{r['E324s_dup']:.2f} | lf {r['E324s_lf']}/{r['E324s_lt']} | ngt {r['E324s_ngt']:.0f}")
for r in rows:
    if r["row"] == 1631 or (r["scene"] == 41 and abs(r["E312_topo"] - 0.82) < 0.03 and abs(r["B_topo"] - 0.35) < 0.05):
        show(r)
print("scene 41 rows:", [(r['k'], r['row']) for r in rows if r['scene'] == 41][:5], "...")
