import cv2, numpy as np, sys, os
for code in ("avc1", "H264", "X264", "vp09", "VP80"):
    ext = ".webm" if code.startswith("VP") or code == "vp09" else ".mp4"
    p = f"/tmp/t_{code}{ext}"
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*code), 2, (320, 240))
    ok = w.isOpened()
    if ok:
        for i in range(4):
            w.write(np.full((240, 320, 3), i * 60, np.uint8))
        w.release()
    print(code, "opened", ok, os.path.getsize(p) if os.path.exists(p) else None)
