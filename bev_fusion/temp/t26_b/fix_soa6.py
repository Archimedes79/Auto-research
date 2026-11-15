p = "evaluation/soa_maps.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    L = [f"# Map layers against GaussianCaR: {meta.get('run') or 'the reference only'} (ours), GT''', '''    L = [f"# Map layers against GaussianCaR: {meta['run'] + ' (ours)' if meta.get('run') and meta.get('device') != 'none' else 'the reference only'}, GT''')
open(p, "w", encoding="utf-8").write(s)
