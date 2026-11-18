import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e379", "ens_step0.py")
s = open(p, encoding="utf-8").read()
s = s.replace('S = int(sys.argv[1])\n', 'S = int(sys.argv[1])\nSPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"        # val: the 873 frames; train: the every-3 sample of the TRAIN dumps (291 frames)\nEVERY = 3 if SPLIT == "train" else 1\n')
s = s.replace('split="val", every=1, drop=0.0', 'split=SPLIT, every=EVERY, drop=0.0')
s = s.replace('VAL {len(frames)} frames', '{SPLIT.upper()} {len(frames)} frames')
open(p, "w", encoding="utf-8").write(s)
print("patched")
