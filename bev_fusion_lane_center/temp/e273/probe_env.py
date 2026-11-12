import importlib
for m in ("cv2", "imageio", "imageio_ffmpeg", "matplotlib", "PIL", "av", "reportlab", "skvideo"):
    try:
        mod = importlib.import_module(m); print(m, "ok", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "NO", type(e).__name__)
try:
    import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception as e:
    print("ffmpeg exe", e)
