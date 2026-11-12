p = "training/grid/qlinks.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    no = dict(e=torch.full((B, N, N), -6.0), uout=torch.full((B, N, N), -6.0), uin=torch.full((B, N, N), -6.0), dir_raw=torch.zeros(B, 2, N, N),", "    no = dict(e=torch.full((B, N, N), -6.0), flags=torch.full((B, 8, N, N), -4.0), uout=torch.full((B, N, N), -6.0), uin=torch.full((B, N, N), -6.0), dir_raw=torch.zeros(B, 2, N, N),")
open(p, "w", encoding="utf-8").write(s)
