"""scratch: the critic's keep_steps fixes (finding 1 not free on a resume, finding 10 None / bool / numpy ints)"""
import re
P = "/mnt/c/Auto-research/autoresearch/trainkit/"
def sub(path, old, new, count=1):
    s = open(P + path).read()
    assert s.count(old) >= 1, (path, old[:50])
    s = s.replace(old, new) if count == 0 else s.replace(old, new, count)
    open(P + path, "w").write(s)
# config.py: not free on a resume; comment
sub("config.py", '"keep", "keep_steps", "slim_at_end"', '"keep", "slim_at_end"')
sub("config.py", "# () = today's pruning, bit for bit. Free on a resume", "# () or None = today's pruning, bit for bit. NOT free on a resume (a resume without the list would prune the protected checkpoints: check_resume refuses it). Protected checkpoints stay full-size\n                                                 # even with slim_at_end (only the final one is written slim)")
# loop.py: one normalised list
sub("loop.py", "    bad_keep = [s for s in cfg.keep_steps if not (isinstance(s, int) and 0 < s <= cfg.steps and cfg.save_every and s % cfg.save_every == 0)]",
    "    keep_steps = tuple(cfg.keep_steps or ())                    # None (a json null) = no list\n    bad_keep = [s for s in keep_steps if not (isinstance(s, numbers.Integral) and not isinstance(s, bool) and 0 < s <= cfg.steps and cfg.save_every and s % cfg.save_every == 0)]")
s = open(P + "loop.py").read()
s = s.replace("checkpoint.prune_checkpoints(out_dir, cfg.keep, cfg.keep_steps)", "checkpoint.prune_checkpoints(out_dir, cfg.keep, keep_steps)")
s = s.replace("if (cfg.slim_at_end or was_slim) else cfg.keep, cfg.keep_steps)", "if (cfg.slim_at_end or was_slim) else cfg.keep, keep_steps)")
if "import numbers" not in s:
    s = s.replace("import json\n", "import json\nimport numbers\n", 1)
open(P + "loop.py", "w").write(s)
