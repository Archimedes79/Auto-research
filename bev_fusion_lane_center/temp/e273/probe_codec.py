import cv2, sys
for f in sys.argv[1:]:
    c = cv2.VideoCapture(f)
    fourcc = int(c.get(cv2.CAP_PROP_FOURCC))
    print(f.split("/")[-1], "".join(chr((fourcc >> 8 * i) & 255) for i in range(4)), int(c.get(cv2.CAP_PROP_FRAME_COUNT)), "frames", int(c.get(3)), "x", int(c.get(4)), c.get(cv2.CAP_PROP_FPS), "fps")
