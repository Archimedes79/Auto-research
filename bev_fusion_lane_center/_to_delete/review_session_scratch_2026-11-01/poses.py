import numpy as np, json, os, glob
H = os.path.expanduser('~/exp_gcar_b123')
print([os.path.basename(p) for p in glob.glob(H + '/poses*')])
z = np.load(H + '/poses_val.npz')
print({k: z[k].shape for k in z.files})
s = json.load(open('/mnt/c/Auto-research/bev_fusion_lane_center/data/splits.json'))
print({k: (len(v), min(v), max(v)) if isinstance(v, list) and v and isinstance(v[0], int) else type(v).__name__ for k, v in s.items()})
sc = z['scene']
for k in ('val', 'test'):
    if k in s:
        r = np.array(s[k])
        print(k, 'unique scenes via poses_val:', len(np.unique(sc[r])) if r.max() < len(sc) else 'OUT OF RANGE')
