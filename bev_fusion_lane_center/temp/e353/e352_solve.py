import os, time, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)
torch.manual_seed(0)
K = 512
for B in (1, 4):
    Q = torch.rand(B, K, K) * 0.002
    A = torch.eye(K)[None] - Q
    s = torch.rand(B, K) * 0.5 + 0.1
    for kind in ("diag_embed rhs", "contiguous eye rhs", "solve vs lu_solve"):
        t0 = time.time()
        if kind == "diag_embed rhs":
            M = torch.linalg.solve(A, torch.diag_embed(s))
        elif kind == "contiguous eye rhs":
            M = torch.linalg.solve(A, (torch.eye(K)[None] * s[:, None, :]).contiguous())
        else:
            LU, piv = torch.linalg.lu_factor(A)
            M = torch.linalg.lu_solve(LU, piv, torch.diag_embed(s))
        print(B, kind, round(time.time() - t0, 2), "s", float(M.abs().mean()), flush=True)
