"""scratch: write one frame of a video to a png. usage: grab_frame.py video.mp4 frame_index out.png"""
import sys
import cv2

cap = cv2.VideoCapture(sys.argv[1])
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(sys.argv[2]), n - 1))
ok, fr = cap.read()
print("frames", n, "size", fr.shape if ok else None, "fps", cap.get(cv2.CAP_PROP_FPS))
if ok:
    cv2.imwrite(sys.argv[3], fr)
