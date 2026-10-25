import importlib.util, shutil, subprocess
for m in ("cv2", "imageio", "imageio_ffmpeg", "matplotlib", "PIL", "moviepy", "av", "numba", "scipy", "torch"):
    print(m, bool(importlib.util.find_spec(m)))
print("ffmpeg:", shutil.which("ffmpeg"))
