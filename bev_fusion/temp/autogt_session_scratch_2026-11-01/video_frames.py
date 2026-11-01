"""scratch: a contact sheet of 4 frames of a crosscheck video (start, 1/3, 2/3, end) -> temp/crosscheck_<scene>.png"""
import sys

import cv2
import numpy as np

path, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(path, "frames", n, "size", int(cap.get(3)), int(cap.get(4)), "fps", cap.get(cv2.CAP_PROP_FPS))
tiles = []
for i in (int(n * 0.05), int(n * 0.35), int(n * 0.65), int(n * 0.95)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    assert ok, i
    tiles.append(cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2)))
sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite(out, sheet)
print("wrote", out, sheet.shape)
