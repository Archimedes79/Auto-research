"""Loop 17: a loop-16 builder on the failure series of p17_series.py (VAL scenes; first half of every scene clean, second half perturbed), per phase.
Writes runs/e17/<out>/<pert>_<variant>.pkl (graphs, gts, rows, scene, phase, per-frame metric components for scene bootstraps) and prints one line per phase.
Usage (gcar, GPU): p17_eval.py --out e130_champion --perts dark,blur,cam_front3_off --variants single --name e123_D7_r3_f1 --set M=288 K0=144 --tau 0.7
  --asm cov_r=0.5 vtx_r=1.5 junc_thr=0.5 reloc_thr=0.4 reloc_shift=12 reloc_tol=1.6   (the champion settings; nothing is selected on the failure series)"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p11_eval as E  # noqa: E402
import p16_common as C  # noqa: E402
import p16_data as PD  # noqa: E402
import p16_infer as PI  # noqa: E402
import p16_train as PT  # noqa: E402

HOME = os.path.expanduser("~")
OUTD = f"{HOME}/exp_gcar_b123/p17"
E17 = "/mnt/c/ML-Experiment/bev_fusion_lane_center/runs/e17"
CHAMP_ASM = dict(cov_r=0.5, vtx_r=1.5, junc_thr=0.5, reloc_thr=0.4, reloc_shift=12, reloc_tol=1.6)


class SeriesData(PD.BuilderData):
    """the frames of one failure series (pert, variant) as builder items; the previous frame of the time recursion is the previous frame of the SERIES (1.5 s earlier), fused like the frame itself"""

    def __init__(self, tag, with_prev=True):
        super().__init__("val_clean", ks=None, train=False, with_prev=False)
        z = np.load(f"{OUTD}/{tag}.meta.npz")
        self.meta = {k: z[k] for k in z.files}
        self.mm17 = np.load(f"{OUTD}/{tag}.npy", mmap_mode="r")
        self.ks = np.array([self.row2k[int(r)] for r in self.meta["rows"]])
        self.sidx = {int(r): i for i, r in enumerate(self.meta["rows"])}
        self.prev17 = with_prev
        self.P17 = dict(np.load(f"{PD.D}/poses_val.npz"))

    def _x_of(self, k):
        return np.asarray(self.mm17[self.sidx[int(self.rows[k])]], np.float16)

    def __getitem__(self, i):
        o = super().__getitem__(i)
        if self.prev17:
            P = self.P17
            row = int(self.rows[int(self.ks[i])])
            pi = int(self.meta["prev"][i])
            o["has_prev"] = torch.tensor(1.0 if pi >= 0 else 0.0)
            o["x_prev"] = torch.from_numpy(np.asarray(self.mm17[pi], np.float16)) if pi >= 0 else torch.zeros_like(o["x"])
            o["Rc"], o["tc"] = torch.from_numpy(P["R"][row].astype(np.float32)), torch.from_numpy(P["t"][row].astype(np.float32))
            prow = int(self.meta["rows"][pi]) if pi >= 0 else row
            o["Rp"], o["tp"] = torch.from_numpy(P["R"][prow].astype(np.float32)), torch.from_numpy(P["t"][prow].astype(np.float32))
        return o


def load_net(design, name, sets):
    c = json.load(open(f"{C.LADDER}/cards/{name}.json", encoding="utf-8-sig"))
    extra = dict(c["cfg"])
    extra.update(PT.parse_set(sets))
    net, cfg = PT.build(design, extra)
    net = net.cuda()
    net.load_state_dict(torch.load(f"{C.OUT}/e123/{name}/model.pt", map_location="cuda"), strict=False)
    net.eval()
    return net, cfg


def summarise(graphs, gts, mask):
    idx = np.where(mask)[0]
    return C.card([graphs[i] for i in idx], [gts[i] for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="D7")
    ap.add_argument("--name", default="e123_D7_r3_f1")
    ap.add_argument("--ckpt", default=None, help="checkpoint dir name under runs/e123 or a full path to model.pt (default: --name)")
    ap.add_argument("--set", nargs="*", default=["M=288", "K0=144"])
    ap.add_argument("--asm", nargs="*", default=[f"{k}={v}" for k, v in CHAMP_ASM.items()])
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--perts", default="dark,blur,cam_front3_off")
    ap.add_argument("--variants", default="single")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--mapnet", type=int, default=0, help="1: the checkpoint is a map-conditioned builder (p17_mapnet.py); needs --ckpt")
    ap.add_argument("--map_mode", default="exact", help="exact | none | wrong (with --mapnet)")
    ap.add_argument("--map_sigma", type=float, default=0.0, help="scene-constant SD map pose error in m (with --mapnet, mode exact)")
    ap.add_argument("--reg", default="", help="loop 18: directory with the registration output (p18_regis_eval.py): the SD map is drawn with the accumulated pose correction and gated by the map trust (needs --mapnet)")
    ap.add_argument("--reg_trust", default="hard", help="none | hard (0 / 1 at --reg_thr) | soft")
    ap.add_argument("--reg_thr", type=float, default=0.62)
    ap.add_argument("--reg_correct", type=int, default=1)
    ap.add_argument("--suffix", default="", help="appended to the result file name")
    ap.add_argument("--prev", type=int, default=1, help="1: time recursion on (needs a builder with cfg.temporal); the previous frame is the previous frame of the series")
    a = ap.parse_args()
    asm = {k: float(v) for k, v in (x.split("=") for x in a.asm)}
    net, cfg = load_net(a.design, a.name, a.set)
    if a.mapnet:
        import p17_mapnet as MN

        MN.widen(net)
    if a.ckpt:
        p = a.ckpt if a.ckpt.endswith(".pt") else f"{C.OUT}/e123/{a.ckpt}/model.pt"
        net.load_state_dict(torch.load(p, map_location="cuda"), strict=False)
    if a.mapnet:
        a.prev = 0
    if not a.prev:
        cfg.temporal = False
        net.cfg.temporal = False
    os.makedirs(f"{E17}/{a.out}", exist_ok=True)
    t0 = time.time()
    for pn in a.perts.split(","):
        for vs in a.variants.split(","):
            tag = f"{pn}_{vs.replace(':', '_')}"
            ds = SeriesData(tag, with_prev=bool(cfg.temporal))
            if a.mapnet:
                if a.reg:
                    import p18_regmap as RM

                    ds = RM.RegMapWrap(ds, "val", ds.P17, a.reg, tag, sigma=a.map_sigma, mode=a.map_mode, trust_mode=a.reg_trust, thr=a.reg_thr, correct=bool(a.reg_correct))
                else:
                    ds = MN.MapWrap(ds, "val", ds.P17, train=False, mode=a.map_mode, sigma=a.map_sigma)
            raw = []
            _, gts, _ = PI.predict(net, ds, list(range(len(ds))), taus=(0.5,), bs=a.bs, workers=2, raw=raw)
            graphs = [PI.assemble(r, None, a.tau, cfg, **asm) for r in raw]
            comp = E.per_sample(graphs, gts)
            ph = ds.meta["phase"]
            tag = tag + a.suffix
            res = dict(tag=tag, name=a.name, tau=a.tau, asm=asm, rows=ds.meta["rows"], scene=ds.meta["scene"], phase=ph, flag=ds.meta["flag"], graphs=graphs, gts=gts, comp=comp, temporal=bool(cfg.temporal))
            cards = {}
            for nm, m in (("cl", ph == 0), ("fail", ph == 1)):
                cards[nm] = summarise(graphs, gts, m)
            res["cards"] = cards
            pickle.dump(res, open(f"{E17}/{a.out}/{tag}.pkl", "wb"))
            print(f"[{tag}] n {len(gts)} | " + " | ".join(f"{nm}: S {c['S']:.3f} F1@1 {c['F1@1.0']:.3f} J {c['J']:.3f} v100 {c['vert100']:.1f} found {c['lanes_found']:.2f}" for nm, c in cards.items()) + f"  {time.time() - t0:.0f} s", flush=True)
    print("DONE", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
