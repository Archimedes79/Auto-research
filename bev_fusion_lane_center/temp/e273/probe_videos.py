import cv2, glob, os, sys
R = os.environ["BFLC_ROOT"]
fs = ["runs/prototype/recall_sp0_dp_video/recall_sp0_dp_val.mp4", "runs/prototype/recall_sp0_dp_junctions/recall_sp0_dp_val.mp4", "reports/weekly/Lane_graph_recall_val.mp4",
      "runs/grid_view/e272_j_s0/scene23.mp4", "runs/grid_view/assign_side/scene7.mp4", "reports/results/week3/video/Week3_lane_prototype_recall_sp0_dp.mp4"]
for f in fs:
    c = cv2.VideoCapture(os.path.join(R, f))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fc = int(c.get(cv2.CAP_PROP_FOURCC))
    print(f, n, "frames", int(c.get(3)), "x", int(c.get(4)), c.get(cv2.CAP_PROP_FPS), "fps", "".join(chr((fc >> 8 * i) & 255) for i in range(4)), round(os.path.getsize(os.path.join(R, f)) / 1e6, 1), "MB", round(os.path.getmtime(os.path.join(R, f))))
    c.set(cv2.CAP_PROP_POS_FRAMES, min(30, n // 2)); ok, fr = c.read()
    if ok:
        cv2.imwrite(os.path.join(R, "temp", "e273", "report", "probe_" + os.path.basename(f).replace(".mp4", ".png")), fr)
