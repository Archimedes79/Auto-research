p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''bootstrap interval {diff, lo, hi, n_valid, ours, ref, thr_ours, thr_ref}; without `ours` (a reference-only run) only the reference's numbers"""''',
              '''bootstrap interval {diff, lo, hi, n_valid, ours, ref}; best = {system: {iou, thr}} each system's best threshold (the d2 numbers); without `ours` (a reference-only run) only the reference's numbers"""''')
s = s.replace('''assert w["ours"] == 0.0 and w["d2"]["thr_ours"] == 0.3 and abs(w["d2"]["ours"] - 1.0) < 1e-12 and w["d1"]["diff"] == -1.0 and abs(w["d2"]["diff"]) < 1e-12, (w["d1"], w["d2"])''',
              '''assert w["ours"] == 0.0 and w["best"]["ours"]["thr"] == 0.3 and abs(w["d2"]["ours"] - 1.0) < 1e-12 and w["d1"]["diff"] == -1.0 and abs(w["d2"]["diff"]) < 1e-12, (w["d1"], w["d2"], w["best"])''')
open(p, "w", encoding="utf-8").write(s)
