p = "bev_fusion_lane_center/evaluation/jtoken_eval.py"
s = open(p, encoding="utf-8").read()
a = """        return row

    rows = pmap(cell_row, SWEEP_GRIDS[a.grid], getattr(a, "jobs", 1))                          # --jobs N: the cells are independent, N forked processes, the same numbers in the same order
    for row in rows:
        lt, ltl, kt, gate = row["link_thr"], row["link_thr_long"], row["keep_thr"], row["none_gate"]
        print(f"link {lt:<5g} long {ltl:<5g} keep {kt:<4g} gate {'on ' if gate else 'off'}  " + "  ".join(f"{k} {row[k]:.4f}" for k in SWEEP_KEYS + JUNCTION_KEYS)
              + f"  links/frame {row['links_per_frame']:.1f}", flush=True)
"""
b = """        print(f"link {lt:<5g} long {ltl:<5g} keep {kt:<4g} gate {'on ' if gate else 'off'}  " + "  ".join(f"{k} {row[k]:.4f}" for k in SWEEP_KEYS + JUNCTION_KEYS)
              + f"  links/frame {row['links_per_frame']:.1f}", flush=True)                   # the progress line of the cell (with --jobs N the lines of different cells interleave; the table is in grid order)
        return row

    rows = pmap(cell_row, SWEEP_GRIDS[a.grid], getattr(a, "jobs", 1))                          # --jobs N: the cells are independent, N forked processes, the same numbers in the same order
"""
assert s.count(a) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
t = "bev_fusion_lane_center/evaluation/tests/test_jtoken_eval.py"
s = open(t, encoding="utf-8").read()
k = "        assert all(np.isfinite(r[k]) for r in prow for k in E.JUNCTION_KEYS + (\"junction_points_per_frame\", \"long_links_per_frame\")), \"the junction numbers and the long links of every row\"\n"
assert s.count(k) == 1
add = k + """        out3 = os.path.join(tmp, "out_jobs")                                                               # --jobs 3: the same table, the same order (forked cells; jobs 1 is the call above)
        os.makedirs(out3)
        pj = json.load(open(E.sweep(E.parse_args(["sweep", "--run", "h", "--seed", "0", "--split", "val", "--max", "3", "--jobs", "3"]), runs, out3)))
        assert pj["rows"] == prow, "sweep --jobs 3 gives the rows of the serial sweep, to the last digit"
"""
open(t, "w", encoding="utf-8", newline="\n").write(s.replace(k, add))
