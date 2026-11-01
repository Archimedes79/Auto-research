p = "autogt/video.py"
s = open(p, encoding="utf-8").read()
s = s.replace("Codec mp4v (desktop players; not every browser).", "H.264 (autogt/encode.py; read back after encoding), 12 fps.")
s = s.replace("import math\nimport os\n\nimport cv2", "import math\nimport os\nimport tempfile\n\nimport cv2")
s = s.replace("from .grid import GridSpec\n", "from .encode import encode\nfrom .grid import GridSpec\n")
a = s.index("    writer = None\n")
b = s.index("    writer.release()")
loop = s[a:b]
loop = loop.replace("    writer = None\n", "    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out_path)))\n    paths = []\n")
loop = loop.replace("""        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
""", """        paths.append(os.path.join(tmp, f"f{n:05d}.png"))
        cv2.imwrite(paths[-1], frame)
""")
tail = s[b:]
tail = tail.replace("""    writer.release()
    print("saved", out_path, f"{len(frames)} frames, {len(frames) / 12:.1f} s")""",
"""    res = encode(paths, out_path, fps=12)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("saved", res["path"], f"{res['frames']} frames, {res['frames'] / 12:.1f} s, {res['codec']} via {res['method']}")
    return res""")
s = s[:a] + loop + tail
open(p, "w", encoding="utf-8").write(s)
print("ok")
