# read-only: which of our 82 val scenes / their samples are in CRN's val infos, and have their radar PV files (no CRN code is run)
import json, os, pickle, glob
home = os.path.expanduser("~")
infos = pickle.load(open(f"{home}/data/nuscenes_full/nuscenes_infos_val_crn.pkl", "rb"))
print("val infos:", len(infos), "keys:", sorted(infos[0].keys())[:12])
split = json.load(open("splits/large.json"))
val = set(split["val"]) if isinstance(split, dict) else set()
print("our val scenes:", len(val))
mine = [i for i in infos if i["scene_name"] in val]
print("samples of our val scenes in the CRN infos:", len(mine), "scenes:", len({i["scene_name"] for i in mine}))
pv = f"{home}/data/nuscenes_full/radar_pv_filter"
have = set(os.listdir(pv))
ok = 0
for i in mine:
    names = [os.path.split(c["filename"])[-1] + ".bin" for c in i["cam_infos"].values()]
    ok += all(n in have for n in names)
print("of them with the radar PV files of all six cameras:", ok)
bev = f"{home}/data/nuscenes_full/radar_bev_filter"
print("radar_bev files:", len(os.listdir(bev)))
c = infos[0]
print("cam_sweeps:", len(c["cam_sweeps"]), "lidar_sweeps:", len(c["lidar_sweeps"]), "lidar file:", c["lidar_infos"]["LIDAR_TOP"]["filename"])
# camera files present for the first sample of our val set (all six cameras, key frame)
root = f"{home}/data/nuscenes_full"
s = mine[0]
print({k: os.path.exists(os.path.join(root, v["filename"])) for k, v in s["cam_infos"].items()})
sw = s["cam_sweeps"][2] if len(s["cam_sweeps"]) > 2 else {}
print("sweep cam files present:", {k: os.path.exists(os.path.join(root, v["filename"])) for k, v in sw.items()})
