p = "temp/gt2/snap_score.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import snapconn as SC\n", "")
s = s.replace('    if str(path).endswith("ends_card.json") and "w" in mode:', '    if "ends_card" in str(path) and "w" in mode:')
s = s.replace("QE.add_connectors = SC.add_connectors_snap\n", "")
s = s.replace("skip_base_check=True, chord=False))", "skip_base_check=True, chord=False, snap=True))")
s = s.replace('sys.path.insert(0, os.path.join(ROOT, "temp", "jr"))\n', "")
open(p, "w", encoding="utf-8").write(s)
print(s[-700:])
