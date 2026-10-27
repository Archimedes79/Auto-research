import json,statistics as st
def L(n): return json.load(open(f'runs/eval/{n}/val.json'))['metrics']
M={(a,s):L(f'e24_j1{a}_s{s}') for a in ('rc','eq') for s in range(4)}
for n in (3,4):
    for k in ('Q','lanes_found','R@1_far'):
        d=[M[('rc',s)][k]-M[('eq',s)][k] for s in range(n)]
        print(n,k,'mean %.5f sd %.5f ratio %.3f'%(st.mean(d),st.stdev(d),st.mean(d)/st.stdev(d)),'rcsd %.5f eqsd %.5f'%(st.stdev([M[('rc',s)][k] for s in range(n)]),st.stdev([M[('eq',s)][k] for s in range(n)])))
# equal-vertex Q check using pair vertex gaps
d=[M[('rc',s)]['Q']-M[('eq',s)]['Q'] for s in range(4)]
v=[M[('rc',s)]['vert100']-M[('eq',s)]['vert100'] for s in range(4)]
print(d,v)
# mean of fast-vs-pair ratios used: 0.0726/0.0043
print(0.07261/st.stdev(d))
