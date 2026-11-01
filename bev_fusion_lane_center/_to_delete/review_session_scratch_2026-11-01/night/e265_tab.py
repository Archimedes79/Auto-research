import json,os,sys
names=["recall_s1","e24_j1rc_s1","e24_j1rc_s2","e24_j1rc_s3","recall","fast_recall","champion","fast_recall_na"]
keys=["Q","lanes_found","R@1_far","topo_F1","false_len_frac","dup_frac","P@1_far","vert100"]
def flat(d):
    m=d.get("metrics",d)
    return m
for n in names:
    for tag in ["l22","sp0","sp25"]:
        p=f"runs/eval/{n}/val_{tag}.json"
        if not os.path.exists(p): continue
        d=json.load(open(p)); m=flat(d)
        row=[]
        for k in keys:
            v=m.get(k)
            row.append(f"{v:.4f}" if isinstance(v,(int,float)) else str(v))
        print(f"{n:15s} {tag:5s} "+" ".join(row))
print("keys:",keys)
d=json.load(open("runs/eval/recall_s1/val_sp0.json")); print(list(d.keys())[:30])
