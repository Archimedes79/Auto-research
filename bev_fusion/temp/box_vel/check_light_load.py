"""t11 vel cause: probe_features.light_load gives the same weights as evaluation/bev_adapter.load_run (EMA applied), every tensor of
the model's state dict bit for bit. Run (from bev_fusion/): OMP_NUM_THREADS=2 ~/miniconda3/envs/gcar/bin/python temp/box_vel/check_light_load.py runs/t8_ctl_s0
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.abspath(os.path.join(HERE, "..", "..")), HERE]


def main(run):
    import torch
    torch.set_num_threads(2)
    from evaluation.bev_adapter import _chain, load_run, setup_paths
    setup_paths(run)
    ns = _chain()
    from probe_features import light_load
    _, m1, l1, s1, c1 = load_run(run, "cpu", None, ns)
    a = {k: v.clone() for k, v in m1.state_dict().items()}
    la = {k: v.clone() for k, v in l1.state_dict().items()}
    del m1, l1
    _, m2, l2, s2, c2 = light_load(run, ns.T, "cpu")
    b, lb = m2.state_dict(), l2.state_dict()
    assert set(a) == set(b) and c1 == c2 and s1 == s2, (len(a), len(b), c1, c2, s1, s2)
    bad = [k for k in a if not torch.equal(a[k], b[k])]
    badl = [k for k in la if not torch.equal(la[k], lb[k])]
    assert not bad and not badl, (bad[:5], badl[:5])
    print(f"light_load ok: {len(a)} model tensors and {len(la)} loss tensors equal to load_run's (EMA applied), step {s1}, {os.path.basename(c1)}")


if __name__ == "__main__":
    main(sys.argv[1])
