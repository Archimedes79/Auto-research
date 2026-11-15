import os, cv2
R = os.environ["BFLC_ROOT"]
im = cv2.imread(os.path.join(R, "results/E279.w_s0/frames/11/scene-0221_e279_w_s0_f19_5dd954d0.png"))
print(im.shape)
c = im[60:475, 655:1075]            # the 1 m centre-cell panel
cv2.imwrite(os.path.join(R, "temp/e284/panel_centre.png"), cv2.resize(c, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST))
c2 = im[60:475, 1500:1920]
cv2.imwrite(os.path.join(R, "temp/e284/panel_nodes.png"), cv2.resize(c2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
