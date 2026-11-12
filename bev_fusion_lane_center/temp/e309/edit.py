import re
p = "temp/e309/perframe.py"
s = open(p).read()
s = s.replace('RUNS = {"e284": f"e284_h2_s{seed}", "e287p2": f"e287_p2_s{seed}"}', 'RUNS = {"e284": f"e284_h2_s{seed}", "e287p2": f"e287_p2_s{seed}", "e312": f"e312_p_s{seed}"}\nTHR = {"e312": 0.1}\nSNAP = {"e312": True}')
s = s.replace("ia, ib = np.nonzero(p >= 0.5)", "ia, ib = np.nonzero(p >= THR.get(k, 0.5))")
s = s.replace("bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs)", "bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), pairs, snap=SNAP.get(k, False))")
open(p, "w").write(s)
