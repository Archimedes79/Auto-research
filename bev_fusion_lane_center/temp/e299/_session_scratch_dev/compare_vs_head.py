"""one-time check before the install (E299 steps A + B): the scratch copy of jends.py / lane_gt_data.py against the COMMITTED files (HEAD copies in _ref/), CPU.
 - the models of E312 (runs/e312_p_s0), E320 (e320_a_s0), E324 s (e324_s_s0): the JEnds weights `je.*` loaded into the HEAD class, the new class flag off and the new class flag on with the gate at 0 (+ a state):
   outputs, jends_loss (+ info) and EVERY gradient (the head's parameters, the heat / feature inputs of the maps head) must be bit-identical; on a synthetic batch AND on real VAL frames (junction GT, geometry loss, the node head of B for the attach models)
 - the loader: the flag-off batches of the scratch lane_gt_data are those of HEAD's (every key, bit for bit)
 - the self-tests of jends.py and jends_eval.py run against the scratch modules
Run: wsl bash tools/env/_py14.sh temp/e299/_session_scratch_dev/compare_vs_head.py"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(HERE, "training", "data"))
sys.path.insert(0, os.path.join(HERE, "training", "grid"))
for s in ("training/grid", "training/data", "training/data/prep", "training/train", "evaluation", ".."):
    sys.path.append(os.path.join(ROOT, s))
import jends as JE  # noqa: E402  (the scratch copy)
import lane_gt_data as LD  # noqa: E402  (the scratch copy)
import jstate as JS  # noqa: E402


def load_ref(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


JER = load_ref("jends_ref", os.path.join(HERE, "_ref", "jends_ref.py"))
LDR = load_ref("lane_gt_data_ref", os.path.join(HERE, "_ref", "lane_gt_data_ref.py"))
assert JE.__file__.startswith(HERE) and LD.__file__.startswith(HERE)
torch.set_num_threads(2)
RUNS = ("e312_p_s0", "e320_a_s0", "e324_s_s0")


def eq(a, b, path=""):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and bool(((a == b) | (a != a) & (b != b)).all()), f"differs: {path}"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (path, set(a) ^ set(b))
        for k in a:
            eq(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            eq(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"differs: {path}: {a} vs {b}"


def synthetic(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    jout = dict(j1=torch.randn(B, 8, 100, 100, generator=g) * 0.5 - 2.5, j05=torch.randn(B, 2, 200, 200, generator=g) - 2.0, f1=torch.randn(B, 64, 100, 100, generator=g))
    P = dict(centre1=torch.randn(B, 12, 100, 100, generator=g))
    ent = torch.tensor([[[-1.75, -8.0, 0.0, 1.0], [1.75, -8.0, 0.0, 1.0]]] * B)
    ext = torch.tensor([[[-1.75, 8.0, 0.0, 1.0], [1.75, 8.0, 0.0, 1.0]]] * B)
    gt = dict(ent=ent, ext=ext, ent_m=torch.ones(B, 2, dtype=torch.bool), ext_m=torch.ones(B, 2, dtype=torch.bool))
    ppts = torch.zeros(B, 2, 16, 2)
    for k_, x_ in enumerate((-1.75, 1.75)):
        ppts[:, k_, :, 0], ppts[:, k_, :, 1] = x_, torch.linspace(-8.0, 8.0, 16)
    batch = dict(ent=ent, ext=ext, ent_m=gt["ent_m"], ext_m=gt["ext_m"], pairs=torch.tensor([[[0, 0], [1, 1]]] * B), pscale=torch.full((B, 2, 2), 0.3), ppts=ppts, pdrv=torch.ones(B, 2))
    cc = -50.0 + (torch.arange(50).float() + 0.5) * 2.0
    nodes = dict(e=torch.full((B, 50, 50), -8.0), flags=torch.full((B, 8, 50, 50), -8.0), uout=torch.full((B, 50, 50), -8.0), uin=torch.full((B, 50, 50), -8.0), s=torch.full((B, 50, 50), -8.0),
                 pos=torch.stack([cc.view(50, 1).expand(50, 50), cc.view(1, 50).expand(50, 50)], 0)[None].expand(B, -1, -1, -1).clone(), dir_raw=torch.zeros(B, 2, 50, 50), tok=torch.randn(B, 64, 50, 50, generator=g))
    nodes["dir_raw"][:, 1] = 1.0
    for (i_, j_) in ((24, 21), (22, 21), (24, 29)):
        nodes["e"][:, i_, j_] = 4.0
    return jout, P, gt, batch, nodes


def run_one(mod, je, jout, P, gt, batch, nodes, geom_w, state=None, tok=None):
    """forward + loss + backward on leaf copies of the inputs of the head -> (out, loss, info, parameter grads, input grads)"""
    je.zero_grad()
    leaves = dict(j1=jout["j1"].detach().clone().requires_grad_(True), j05=jout["j05"].detach().clone().requires_grad_(True), f1=jout["f1"].detach().clone().requires_grad_(True))
    Pl = dict(centre1=P["centre1"].detach().clone().requires_grad_(True))
    kw = dict(gt=gt, tok=tok, nodes=nodes if je.attach is not None else None)
    out = je(leaves, Pl, **kw, **({} if state is None else dict(state=state)))
    loss, info = mod.jends_loss(out, batch, geom_w=geom_w)
    loss.backward()
    pg = {n: (None if p.grad is None else p.grad.clone()) for n, p in je.named_parameters() if not n.startswith("state_attn")}
    ig = {k: (None if v.grad is None else v.grad.clone()) for k, v in {**leaves, "centre1": Pl["centre1"]}.items()}
    return dict(out={k: v.detach() for k, v in out.items()}, loss=loss.detach(), info=info, pg=pg, ig=ig)


def compare(tag, jout, P, gt, batch, nodes, tok, ck, args, geom_w):
    kw = dict(d=64, order=not args["no_order"], evidence=not args["no_evidence"], f1=args["maps_args"].get("d", 64), scales=bool(args.get("scales", False)), ntok=64 if args.get("ntok") else 0, unc=bool(args.get("unc", False)), attach=bool(args.get("attach", False)))
    st = {k[3:]: v for k, v in ck["state"].items() if k.startswith("je.")}
    ref, new, on = JER.JEnds(**kw), JE.JEnds(**kw), JE.JEnds(**kw, state=True)
    ref.load_state_dict(st)
    new.load_state_dict(st)
    rep = on.load_state_dict(st, strict=False)
    assert not rep.unexpected_keys and all(k.startswith("state_attn.") for k in rep.missing_keys), rep
    for m in (ref, new, on):
        m.eval()
    r = run_one(JER, ref, jout, P, gt, batch, nodes, geom_w, tok=tok)
    n = run_one(JE, new, jout, P, gt, batch, nodes, geom_w, tok=tok)
    eq(r, n, tag + " flag off")
    with torch.no_grad():
        so = JS.build_state(new(jout, P, gt=None, tok=tok, nodes=nodes if new.attach is not None else None))
    on_r = run_one(JE, on, jout, P, gt, batch, nodes, geom_w, state=JS.warp_state(so, torch.eye(2).expand(jout["j1"].shape[0], 2, 2), torch.zeros(jout["j1"].shape[0], 2)), tok=tok)
    eq(r, on_r, tag + " flag on, gate 0, with a state")
    n_par = sum(v is not None for v in r["pg"].values())
    print(f"  {tag}: outputs ({len(r['out'])} keys), loss {float(r['loss']):.6f}, {n_par} parameter gradients, input gradients: bit-identical (HEAD = flag off = flag on at gate 0)")


def main():
    # 1. the self-tests against the scratch modules
    JE._selftest()
    import jends_eval as JEV
    JEV.selftest()
    # 2. the flag-off batches of the loader
    for cls in (LD, LDR):
        cls.QData.DEFAULT_JUNCTION, cls.QData.DEFAULT_PSCALE, cls.QData.DEFAULT_PPTS = True, True, True
    dsn, dln = LD.loader("val_clean", bs=2, workers=0, shuffle=False, map_mode="exact", pixels=False)
    dsr, dlr = LDR.loader("val_clean", bs=2, workers=0, shuffle=False, map_mode="exact", pixels=False)
    batches = []
    for i, (bn, br) in enumerate(zip(dln, dlr)):
        eq(bn, br, "batch %d" % i)
        batches.append(bn)
        if i == 1:
            break
    print("loader flag off: 2 batches of 2 real VAL frames, every key bit-identical to HEAD's lane_gt_data")
    # 3. the models
    for run in RUNS:
        ck = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")
        args = ck["args"]
        geom_w = float(args.get("geom_w", 0.0))
        import jends_train as JT
        model = JT.JEndModel(args["maps_args"], order=not args["no_order"], evidence=not args["no_evidence"], scales=bool(args.get("scales", False)), ntok=bool(args.get("ntok", False)), unc=bool(args.get("unc", False)), attach=bool(args.get("attach", False)))
        model.load_state_dict(ck["state"])
        model.eval()
        print(run, "geom_w", geom_w, "attach", bool(args.get("attach")), "scales", bool(args.get("scales")))
        jout, P, gt, batch, nodes = synthetic()
        compare(run + " synthetic", jout, P, gt, batch, nodes if args.get("attach") else None, None, ck, args, geom_w)
        for bi, b in enumerate(batches):
            with torch.no_grad():
                Pm, out = model.jmodel(b["x"].float(), joint=False)
                nd = model.nodes(Pm["feat2"], Pm["feat1"], Pm["feat05"], Pm["road2"], Pm["centre1"]) if model.nodes is not None else None
            gtb = {k: b[k] for k in ("ent", "ext", "ent_m", "ext_m")}
            compare(f"{run} real batch {bi}", out, Pm, gtb, b, nd if args.get("attach") else None, nd["tok"] if args.get("ntok") else None, ck, args, geom_w)
    print("compare_vs_head ok")


if __name__ == "__main__":
    main()
