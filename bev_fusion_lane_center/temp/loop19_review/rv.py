import json, glob, os, numpy as np
R=os.environ.get("BFLC_ROOT",".")+"/runs/eval"
ks=("Q","lanes_found","R@1_far","topo_F1","F1@1.0","vert100","J")
def L(n,f="val.json"):
    return json.load(open(f"{R}/{n}/{f}",encoding="utf-8-sig"))["metrics"]
print("== E240")
v={}
for arm in("rc","eq"):
    v[arm]={k:np.array([L(f"e24_j1{arm}_s{s}")[k] for s in range(3)]) for k in ks}
    for k in ks:
        a=v[arm][k]; print(arm,k,np.round(a,4).tolist(),"mean %.4f sd %.4f"%(a.mean(),a.std(ddof=1)))
for k in ks:
    d=v["rc"][k]-v["eq"][k]; print("diff",k,np.round(d,4).tolist(),"mean %.4f sd %.4f"%(d.mean(),d.std(ddof=1)))
# pooled SD
for k in ks:
    p=np.sqrt((v["rc"][k].var(ddof=1)+v["eq"][k].var(ddof=1))/2); print("pooled",k,"%.4f"%p)
print("== E241")
for p in("25","50","75"):
    a=L(f"e24_j1rc_s2_at{p}");c=L(f"e24_j1eq_s2_at{p}")
    print(p,{k:(round(a[k],4),round(c[k],4),round(a[k]-c[k],4)) for k in ("lanes_found","Q","R@1_far","vert100")})
a=L("e24_j1rc_s2");c=L("e24_j1eq_s2")
print("final",{k:(round(a[k],4),round(c[k],4),round(a[k]-c[k],4)) for k in ("lanes_found","Q","R@1_far")})
print("== E244")
for n in("e24_j1cl8_s0","e24_j1eq_s0","e24_j1eq_s2_at50","e24_j1rc_s2_at50","e24_j1phd_s0_at50"):
    m=L(n);print(n,{k:round(m[k],4) for k in ("Q","lanes_found","R@1_far","topo_F1","vert100")}, json.load(open(f"{R}/{n}/val.json",encoding="utf-8-sig"))["registry"].get("tau"))
for f in sorted(glob.glob(f"{R}/e24_j1phd_s0_at50/val*.json")):
    d=json.load(open(f,encoding="utf-8-sig"));m=d["metrics"]
    print(os.path.basename(f),d["registry"].get("tau"),d.get("overrides"),{k:round(m[k],4) for k in ("Q","lanes_found","R@1_far","vert100")})
print("== E237")
for n in("e24_j1ex_s0","e24_j1ex_s1","e24_j1eq_s0","e24_j1eq_s1"):
    m=L(n);print(n,{k:round(m[k],4) for k in ("Q","lanes_found","R@1_far","topo_F1","J","F1@1.0")})
