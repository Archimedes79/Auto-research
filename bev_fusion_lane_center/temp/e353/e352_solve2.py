import os, sys, time, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(int(sys.argv[2]))
torch.manual_seed(0)
K = int(sys.argv[1])
Q = torch.rand(1, K, K) * 0.002
A = torch.eye(K)[None] - Q
t0 = time.time()
M = torch.linalg.solve(A, torch.diag_embed(torch.rand(1, K) * 0.5 + 0.1))
print("K", K, "threads", sys.argv[2], round(time.time() - t0, 2), "s", float(M.abs().mean()), flush=True)
