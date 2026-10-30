import glob, sys, time
sys.argv += ["--quick", "--data", "nuscenes", "--sharp", "--pyr", "--doppler", "--rawwarp", "3.0", "--prior", "0.03"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import test_flow_explains as T
T.DEV = "cpu"
t0 = time.time()
files = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))[:40]
items = [it for it in (T.nuscenes_item(f) for f in files) if it is not None]
print(f"{len(items)} of {len(files)} samples have a moving vehicle ({time.time() - t0:.1f} s)")
maps, vgt, bid, s01, s12 = items[0]
print("maps", tuple(maps.shape), "speed max", float(vgt.norm(dim=0).max()), "steps", len(s01), len(s12), s01[0], s12[0])
print("G", T.G.nx, T.G.ny, "TAG", T.TAG)
m = T.Model()
T.train(m, items, 2, __import__("random").Random(0), bs=2)
r = T.evaluate(m.eval(), items[:2])
print({k: round(v, 2) for k, v in r.items()})
T.picture(m, items, "temp/checks/dry_nusc_pic.png", n=2)
print("ok")
