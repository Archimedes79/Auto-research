import cv2, sys, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "Junction.report", "vid"))
for f in sorted(os.listdir(R)):
    c = cv2.VideoCapture(os.path.join(R, f))
    n = 0
    while True:
        ok, fr = c.read()
        if not ok:
            break
        n += 1
    print(f, "frames read:", n, "fps", c.get(5), "size", int(c.get(3)), int(c.get(4)))
print([x for x in dir(cv2) if "FFMPEG" in x][:3])
for cc in ("avc1", "H264", "vp80", "vp09", "mp4v", "MJPG"):
    p = "/tmp/t_%s.%s" % (cc, "webm" if cc.startswith("vp") else "mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*cc), 2, (64, 64))
    print(cc, "opened", w.isOpened())
    w.release()
