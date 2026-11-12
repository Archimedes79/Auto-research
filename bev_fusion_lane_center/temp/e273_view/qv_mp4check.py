import cv2, os, glob
R = os.environ["BFLC_ROOT"]
for p in sorted(glob.glob(os.path.join(R, "temp/e273_view/video/*.mp4"))):
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, fr = cap.read()
    print(os.path.basename(p), n, "frames", w, "x", h, "readable", ok, "kB", os.path.getsize(p) // 1000)
