import cv2, sys
cap = cv2.VideoCapture(sys.argv[1]); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
ok, f = cap.read(); print(ok, n, f.shape if ok else None); cv2.imwrite(sys.argv[2], f)
