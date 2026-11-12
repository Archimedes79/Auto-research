import cv2, sys
cap = cv2.VideoCapture(sys.argv[1]); print("frames", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_POS_FRAMES, 19); ok, fr = cap.read(); print(ok, None if not ok else fr.shape)
if ok: cv2.imwrite(sys.argv[2], fr)
