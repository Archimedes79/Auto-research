"""read the E393 gate arms: per-step cov / nact from model.pt rec, s/step and reserved memory from log.txt, nvidia-smi peak from smi_<arm>.csv"""
import os, re, sys, math, torch, statistics as st
os.chdir(os.environ.get("BFLC_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))   # the problem folder (v1.3: the copy lies in experiments/E393_*/scripts/e393_gate/; the original in temp/e393_gate/ went 2 levels up)
for a in sys.argv[1:]:
    R = f"runs/e393_gate_{a}_s0"
    if not os.path.exists(f"{R}/model.pt"):
        print(a, "no model.pt"); continue
    rec = torch.load(f"{R}/model.pt", map_location="cpu", weights_only=False)["rec"]
    f = lambda v: [x for x in v if not math.isnan(x)]
    cov, na, h, b = f(rec["cov"]), f(rec["nact"]), f(rec["h"]), f(rec["b"])
    log = open(f"{R}/log.txt").read()
    clk = re.findall(r"clock: ([\d.]+) s/step.*?reserved ([\d.]+) GB", log)
    smi = []
    if os.path.exists(f"temp/e393_gate/smi_{a}.csv"):
        smi = [int(l.split(",")[1].split()[0]) for l in open(f"temp/e393_gate/smi_{a}.csv") if "," in l]
    grad = re.findall(r"step 1: .*", log)
    print(f"{a}: steps {len(rec['h'])} overflow {rec['overflow']} | cov first10 {st.mean(cov[:10]):.4f} all {st.mean(cov):.4f} [{min(cov):.4f}, {max(cov):.4f}] | "
          f"nact first10 {st.mean(na[:10]):.1f} all {st.mean(na):.1f} sd {st.pstdev(na):.1f} [{min(na):.1f}, {max(na):.1f}] | head loss first10 {st.mean(h[:10]):.4f} last10 {st.mean(h[-10:]):.4f} | "
          f"b loss first10 {st.mean(b[:10]):.4f} last10 {st.mean(b[-10:]):.4f} | clock {clk} | smi peak {max(smi) if smi else '-'} MiB")
    for g in grad: print("   ", g[:300])
