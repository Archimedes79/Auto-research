import glob, random, sys
sys.argv += ["--quick", "--data", "nuscenes", "--recursive", "--sharp", "--pyr", "--doppler", "--rawwarp", "3.0", "--prior", "0.03", "--smooth", "0.05"]
sys.path.insert(0, "tests"); sys.path.insert(0, "training")
import torch
import test_flow_explains as T
T.DEV = "cpu"
files = sorted(glob.glob("data/cache/gt/*_0.5_4_*.npz"))[:30]
items = [it for it in (T.nuscenes_item(f, True) for f in files) if it is not None][:4]
torch.save(items, "temp/checks/_dry_items.pt"); items = torch.load("temp/checks/_dry_items.pt", weights_only=False)
print(len(items), "items,", len(items[0]), "fields")
m = T.Model()
T.train(m, items, 2, random.Random(0), bs=2)
m.eval()
print({k: round(v, 2) for k, v in T.evaluate(m, items[:2]).items() if k.startswith("EPE vehicles")})
print({k: round(v, 2) for k, v in T.evaluate_recursive(m, items[:2]).items()})
