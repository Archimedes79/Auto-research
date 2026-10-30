import sys, cv2
cap = cv2.VideoCapture(sys.argv[1]); cap.set(cv2.CAP_PROP_POS_FRAMES, int(sys.argv[3])); ok, f = cap.read(); cv2.imwrite(sys.argv[2], f); print(ok, f.shape)
