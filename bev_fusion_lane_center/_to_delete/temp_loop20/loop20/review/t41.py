import json,statistics as st
def L(n):
    d=json.load(open(f'runs/eval/{n}/val.json'))
    assert d['split']=='val' and d['frames']==873,(n,d['split'],d['frames'])
    return d['metrics'],d
M={}
for a in ('rc','eq'):
    for s in range(4):
        M[(a,s)]=L(f'e24_j1{a}_s{s}')[0]
keys=['Q','lanes_found','R@1_far','topo_F1','vert100']
for s in range(4):
    r,e=M[('rc',s)],M[('eq',s)]
    print(s,' '.join(f"{k}: {r[k]:.4f} {e[k]:.4f} {r[k]-e[k]:+.4f}" for k in keys))
for k in keys:
    for a in ('rc','eq'):
        v=[M[(a,s)][k] for s in range(4)]
        print(k,a,'mean %.4f sd1 %.4f sd0 %.4f'%(st.mean(v),st.stdev(v),st.pstdev(v)))
    dv=[M[('rc',s)][k]-M[('eq',s)][k] for s in range(4)]
    print(k,'diff mean %.4f sd1 %.4f sd0 %.4f'%(st.mean(dv),st.stdev(dv),st.pstdev(dv)), 'ratio',st.mean(dv)/st.stdev(dv))
# E241
print('E241')
for s in (2,3):
    for p in (25,50,75):
        r,_=L(f'e24_j1rc_s{s}_at{p}'); e,_=L(f'e24_j1eq_s{s}_at{p}')
        print(s,p,'lanes gap %.4f Q gap %.4f'%(r['lanes_found']-e['lanes_found'],r['Q']-e['Q']), 'verts',round(r['vert100'],2),round(e['vert100'],2))
    r,e=M[('rc',s)],M[('eq',s)]
    print(s,100,'lanes gap %.4f Q gap %.4f'%(r['lanes_found']-e['lanes_found'],r['Q']-e['Q']))
print('cross')
for p in (25,50,75,100):
    for sr in (2,3):
        for se in (2,3):
            def g(a,s):
                return M[(a,s)] if p==100 else L(f'e24_j1{a}_s{s}_at{p}')[0]
            r=g('rc',sr);e=g('eq',se)
            print(p,'rc',sr,'eq',se,'lanes %.3f Q %.4f'%(r['lanes_found']-e['lanes_found'],r['Q']-e['Q']))
