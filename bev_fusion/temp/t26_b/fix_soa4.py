p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()

# rows: every system's best threshold, also for a reference-only run
s = s.replace('''            for s in systems:
                row[s] = iou(total[s][t05, li, ti])
''', '''            for s in systems:
                row[s] = iou(total[s][t05, li, ti])
            row["best"] = {s: {"iou": iou(total[s][best[s][li, ti], li, ti]), "thr": THR[best[s][li, ti]]} for s in systems}
''')
s = s.replace('''                row["d2"] = {**row.get("d2", {}), "thr_ours": THR[best["ours"][li, ti]], "thr_ref": THR[best["ref"][li, ti]]}
''', '')
s = s.replace('''    per layer, tolerance""" ''', '')
# markdown
s = s.replace('''                d2 = r.get("d2", {})
                L.append(f"| {r['layer']} | {r['tol']:g} | {r['n_gt']} | {fmt(r.get('ours'))} | {fmt(r.get('ref'))} | {fmt(r.get('ratio'), 2)} | {interval(r.get('d1'))} | "
                         f"{fmt(d2.get('ours'))} (thr {d2.get('thr_ours', '-')}) | {fmt(d2.get('ref'))} (thr {d2.get('thr_ref', '-')}) | {interval(d2) if 'diff' in d2 else 'n/a'} |")''',
'''                b = r.get("best", {})
                L.append(f"| {r['layer']} | {r['tol']:g} | {r['n_gt']} | {fmt(r.get('ours'))} | {fmt(r.get('ref'))} | {fmt(r.get('ratio'), 2)} | {interval(r.get('d1'))} | "
                         f"{fmt(b.get('ours', {}).get('iou'))} (thr {b.get('ours', {}).get('thr', '-')}) | {fmt(b.get('ref', {}).get('iou'))} (thr {b.get('ref', {}).get('thr', '-')}) | {interval(r.get('d2'))} |")''')
s = s.replace('''    L = [f"# Map layers against GaussianCaR: {meta.get('run')} (ours), GT''', '''    L = [f"# Map layers against GaussianCaR: {meta.get('run') or 'the reference only'} (ours), GT''')
s = s.replace('''            print(f"claim {r['layer']:14s} iou@{r['tol']:g}  n_gt {r['n_gt']:8d}  ours {fmt(r.get('ours'))}  ref {fmt(r.get('ref'))}  d1 {interval(r.get('d1'))}  d2 {interval(r.get('d2') if 'diff' in r.get('d2', {}) else None)}")''',
              '''            print(f"claim {r['layer']:14s} iou@{r['tol']:g}  n_gt {r['n_gt']:8d}  ours {fmt(r.get('ours'))}  ref {fmt(r.get('ref'))}  d1 {interval(r.get('d1'))}  d2 {interval(r.get('d2'))}")''')
# check_stored: table 2 too
s = s.replace('''                        diff = max(diff, abs(want - got))
            out["table1_max_diff"] = diff''', '''                        diff = max(diff, abs(want - got))
                        best = max(v for v in (sys_json[names[s]][layer]["all"][str(th)][tol] for th in THR) if v is not None)
                        diff2 = max(diff2, abs(best - max(iou(scores.cnt[("old", s)].sum(0)[k, li, ti]) for k in range(len(THR)))))
            out["table1_max_diff"], out["table2_max_diff"] = diff, diff2''')
s = s.replace('''            diff = 0.0
            for s in scores.systems("old"):''', '''            diff = diff2 = 0.0
            for s in scores.systems("old"):''')
open(p, "w", encoding="utf-8").write(s)
