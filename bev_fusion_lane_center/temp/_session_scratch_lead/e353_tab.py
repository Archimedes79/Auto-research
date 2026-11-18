import json,os
def g(run,name):
    p=f"runs/{run}/ends_card_{name}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p))["numbers"]["head"]
    def f(d,k):
        if k in d: return d[k]
        for v in d.values():
            if isinstance(v,dict):
                r=f(v,k)
                if r is not None: return r
    return {k:f(d,k) for k in ("topo_F1","F1@1.0","junction_F1","dup_frac","vert100_dp")}
for ro,name in (("snap","snap_thr0.1"),("PRIMARY","snap_nogeom_attach_thr0.1"),("attach","attach_thr0.1"),("unsnapped","thr0.1")):
    for s in (0,1):
        row=[]
        for lab,run in (("c",f"e324_s_s{s}"),("cp",f"e353_cp_s{s}"),("n1",f"e353_n1_s{s}"),("n1z",f"e353_n1z_s{s}")):
            m=g(run,name)
            row.append(f"{lab} {m['topo_F1']:.4f}" if m and m['topo_F1'] is not None else f"{lab} -")
        print(ro,"seed",s,"|"," | ".join(row))
