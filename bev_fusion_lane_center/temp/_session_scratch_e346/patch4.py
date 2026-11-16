p = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e346/e346_lib.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('''    """[nodes, cells, short links, long links, cells with two or more nodes, channels per cell] of one frame''', '''    """[nodes, cells, short links, long links, cells with two or more nodes, channels per cell, candidate nodes dropped (a cell claimed by two pieces)] of one frame''')
rep('''len(cell) - len(np.unique(cell)), ch], float)''', '''len(cell) - len(np.unique(cell)), ch, nt["stats"]["dropped"]], float)''')
rep('''    m["node_cell_share"] = float(o[0] / max(o[1], 1))''', '''    m["node_cell_share"], m["dropped_candidates_per_node"] = float(o[0] / max(o[1], 1)), float(o[6] / max(o[0], 1))''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
