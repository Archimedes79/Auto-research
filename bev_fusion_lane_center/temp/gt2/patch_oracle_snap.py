p = "temp/gt2/oracle_pairs.py"
s = open(p, encoding="utf-8").read()
s = s.replace('run, seed = sys.argv[1], int(sys.argv[2])', 'run, seed = sys.argv[1], int(sys.argv[2])\nSNAP = len(sys.argv) > 3 and sys.argv[3] == "snap"')
s = s.replace('target = os.path.join(ROOT, "temp", "gt2", "thr", f"ORACLE_{run}_thr0.5.json")', 'target = os.path.join(ROOT, "temp", "gt2", "thr", f"ORACLE_{run}_{\'snap_\' if SNAP else \'\'}thr0.5.json")')
s = s.replace('if p.endswith("ends_card.json") and "w" in mode:', 'if "ends_card" in p and "w" in mode:')
s = s.replace("skip_base_check=True, chord=False))", "skip_base_check=True, chord=False, snap=SNAP))")
open(p, "w", encoding="utf-8").write(s)
print("ok")
