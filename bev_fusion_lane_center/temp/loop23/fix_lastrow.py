p = "training/trainer.py"
s = open(p, encoding="utf-8").read()
a = "            if updated and step % log_every == 0:\n"
b = "            if updated and (step % log_every == 0 or step == steps):                 # the last step is always logged: a short run (a smoke) has a row\n"
assert s.count(a) == 1
s = s.replace(a, b)
a = "                if guard and step >= 4 * log_every and os.path.exists(ck):\n"
b = "                if guard and step % log_every == 0 and step >= 4 * log_every and os.path.exists(ck):\n"
assert s.count(a) == 1
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
t = "training/trainer_selftest.py"
u = open(t, encoding="utf-8").read()
a = "# (9) a checkpoint loaded onto the GPU"
b = '''# (8b) the last step is logged even when it is not a multiple of log_every (a 40-step smoke with the default log_every of 100 had no row)
T.fit(make_model(), loss_fn, T.Run("e_last", tmp), **dict(kw, steps=45, log_every=20, ckpt_every=0, guard=False))
steps_logged = [json.loads(ln)["step"] for ln in open(os.path.join(tmp, "e_last", "log.jsonl"))]
assert steps_logged == [20, 40, 45], steps_logged
print("the last step is logged:", steps_logged)

# (9) a checkpoint loaded onto the GPU'''
assert u.count(a) == 1
open(t, "w", encoding="utf-8").write(u.replace(a, b))
print("ok")
