p = "temp/_session_scratch_e291/apply_hunks.py"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if "__import__" in l)
L[i] = '         "    import numpy as np\n"\n         "    arms = JP.collate([dict(c=np.zeros((1, 2), \\"float32\\"), t=np.array([[0.0, 1.0]], \\"float32\\"), n_in=np.array([2]), n_out=np.array([2]))])\n"'
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(L))
