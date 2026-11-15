P = "/mnt/c/Auto-research/autoresearch/trainkit/selftest.py"
s = open(P).read()
a = s.index('        run(tmp, "keepnp"')
b = s.index('        run(tmp, "keepnone"')
s = s[:a] + s[b:]
s = s.replace("# numpy integers are steps too, and a json null is no list", "")
open(P, "w").write(s)
M = "/mnt/c/Auto-research/bev_fusion/temp/cleanup/mut_keep.sh"
m = open(M).read()
m = m.replace("""'s/if (cfg.slim_at_end or was_slim) else cfg.keep, cfg.keep_steps)/if (cfg.slim_at_end or was_slim) else cfg.keep, ())/'""", """'s/if (cfg.slim_at_end or was_slim) else cfg.keep, keep_steps)/if (cfg.slim_at_end or was_slim) else cfg.keep, ())/'""")
m = m.replace("""'0,/checkpoint.prune_checkpoints(out_dir, cfg.keep, cfg.keep_steps)/s//checkpoint.prune_checkpoints(out_dir, cfg.keep)/'""", """'0,/checkpoint.prune_checkpoints(out_dir, cfg.keep, keep_steps)/s//checkpoint.prune_checkpoints(out_dir, cfg.keep)/'""")
m = m.replace("""'s/"keep", "keep_steps", "slim_at_end"/"keep", "slim_at_end"/'""", """'s/"keep", "slim_at_end"/"keep", "keep_steps", "slim_at_end"/'""")
open(M, "w").write(m)
