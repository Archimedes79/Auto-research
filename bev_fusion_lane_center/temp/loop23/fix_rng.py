p = "training/trainer.py"
s = open(p, encoding="utf-8").read()
a = '''def _set_rng(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
'''
b = '''def _set_rng(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"].cpu())                                # a checkpoint loaded with map_location=cuda holds the RNG states on the GPU; set_rng_state wants CPU ByteTensors
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])
'''
assert s.count(a) == 1
open(p, "w", encoding="utf-8").write(s.replace(a, b))
t = "training/trainer_selftest.py"
u = open(t, encoding="utf-8").read()
a2 = 'shutil.rmtree(tmp, ignore_errors=True)\nprint("trainer selftest passed")'
b2 = '''# (9) a checkpoint loaded onto the GPU (map_location=cuda, as fit does on a resume) restores the RNG states (reported 2026-11-03: set_rng_state refused the CUDA tensors, every GPU resume failed)
if torch.cuda.is_available():
    torch.save(T._rng_state(), os.path.join(tmp, "rng.pt"))
    st_gpu = torch.load(os.path.join(tmp, "rng.pt"), map_location="cuda", weights_only=False)
    T._set_rng(st_gpu)
    print("RNG states of a GPU-loaded checkpoint restored:", True)
else:
    print("(9) skipped: no CUDA device here")
shutil.rmtree(tmp, ignore_errors=True)
print("trainer selftest passed")'''
assert u.count(a2) == 1
open(t, "w", encoding="utf-8").write(u.replace(a2, b2))
print("ok")
