"""E381 card v2.1: the verdict by the written rule from the paired cards (CPU, scratch). Writes temp/e381/table381.md and prints the per-seed clauses."""
import json, os
os.chdir(os.environ.get('BFLC_ROOT', '.'))
FLOOR = {0: dict(topo_F1=0.6359, F1=0.8182, lf=0.6066), 1: dict(topo_F1=0.6349, F1=0.8060, lf=0.5937)}
files = {('verdict', 0): 'runs/e381_n_s0/token_card_rec_lt0.3_ltl0.2_kt0.5_nng_vs_e373_cb_s0_rc0.3_0.2_0.6_off.json',
         ('verdict', 1): 'runs/e381_n_s1/token_card_rec_lt0.3_ltl0.2_kt0.5_nng_vs_e373_cb_s1_rc0.3_0.2_0.6_off.json',
         ('d115', 0): 'runs/e381_n_s0/token_card_rec_lt0.3_ltl0.2_kt0.6_nng_vs_e373_cb_s0.json',
         ('d115', 1): 'runs/e381_n_s1/token_card_rec_lt0.3_ltl0.2_kt0.6_nng_vs_e373_cb_s1.json',
         ('own s1 0.2/0.2/0.5', 1): 'runs/e381_n_s1/token_card_rec_lt0.2_kt0.5_nng_vs_e373_cb_s1_rc0.3_0.2_0.6_off.json'}
L = ['| read | seed | n topo F1 | ref topo F1 (D115) | topo F1 n - ref [95 %] | junction F1 | F1@1.0 | lanes found | dup_frac | J | seed clauses |', '|---|---|---|---|---|---|---|---|---|---|---|']
res = {}
for (read, S), f in files.items():
    c = json.load(open(f)); pv = c['info']['paired_vs_ref_run']; m = c['metrics']
    d = lambda k: pv[k]['diff']
    fl = FLOOR[S]
    floors = m['topo_F1'] >= fl['topo_F1'] and m['F1@1.0'] >= fl['F1'] and m['lanes_found'] >= fl['lf']
    adopt = d('topo_F1') >= 0.005 and pv['topo_F1']['lo'] > 0 and d('junction_F1') >= -0.010 and d('F1@1.0') >= -0.005 and d('lanes_found') >= -0.010 and d('dup_frac') <= 0.010 and floors
    no_topo = d('topo_F1') <= -0.005
    guard_broken = d('junction_F1') < -0.010 or d('F1@1.0') < -0.005 or d('lanes_found') < -0.010 or d('dup_frac') > 0.010 or not floors
    res[(read, S)] = dict(adopt=adopt, no_topo=no_topo, guard_broken=guard_broken)
    tag = 'ADOPT clauses hold' if adopt else ('NO-side: topo <= -0.005' if no_topo else ('a guard broken' if guard_broken else 'neither (TIE-side)'))
    L.append(f"| {read} | {S} | {pv['topo_F1']['a']:.4f} | {pv['topo_F1']['b']:.4f} | {d('topo_F1'):+.4f} [{pv['topo_F1']['lo']:+.4f}, {pv['topo_F1']['hi']:+.4f}] | {d('junction_F1'):+.4f} | {d('F1@1.0'):+.4f} | {d('lanes_found'):+.4f} | {d('dup_frac'):+.4f} | {d('J'):+.4f} | {tag}; floors {'hold' if floors else 'BROKEN'} |")
v = [res[('verdict', S)] for S in (0, 1)]
verdict = 'ADOPT-NOISE' if all(x['adopt'] for x in v) else ('NO' if all(x['no_topo'] for x in v) or all(x['guard_broken'] for x in v) else 'TIE')
open('temp/e381/table381.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L)); print('VERDICT (recipe cell 0.3 / 0.2 / 0.5 / off):', verdict, v)
