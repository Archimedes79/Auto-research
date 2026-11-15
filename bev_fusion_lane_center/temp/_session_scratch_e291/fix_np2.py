p = "temp/_session_scratch_e291/apply_hunks.py"
s = open(p, encoding="utf-8").read()
bs = chr(92)
s = s.replace('import numpy as np\n"\n', 'import numpy as np' + bs + 'n"\n')
s = s.replace('n_out=np.array([2]))])\n"\n', 'n_out=np.array([2]))])' + bs + 'n"\n')
open(p, "w", encoding="utf-8", newline="\n").write(s)
