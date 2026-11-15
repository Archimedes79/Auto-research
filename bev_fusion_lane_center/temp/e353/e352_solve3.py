import os, sys, time, torch, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(2)
torch.manual_seed(0)
K, B = 512, 4
how = sys.argv[1]
Q = torch.rand(B, K, K) * 0.002
A = torch.eye(K)[None] - Q
rhs = torch.diag_embed(torch.rand(B, K) * 0.5 + 0.1)
t0 = time.time()
if how == "f64":
    M = torch.linalg.solve(A.double(), rhs.double()).float()
elif how == "lu":
    LU, piv = torch.linalg.lu_factor(A)
    M = torch.linalg.lu_solve(LU, piv, rhs)
elif how == "numpy":
    M = torch.from_numpy(np.linalg.solve(A.numpy(), rhs.numpy()))
elif how == "inv":
    M = torch.linalg.inv(A) @ rhs
elif how == "f32":
    M = torch.linalg.solve(A, rhs)
print(how, round(time.time() - t0, 2), "s", float(M.abs().mean()), float((A @ M - rhs).abs().max()), flush=True)
