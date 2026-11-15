p = "tests/test_student_window.py"
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
rep("ok[f\"a {'older-frame' if older else 'key-frame'} build", "ok[f\"{'an older-frame' if older else 'a key-frame'} build")
rep("""  F  window images on nuScenes mini: each cached older image is the JPEG of its frame's sample_data (independent walk along `prev`), the resumable builder and its finished-build marker (scenes,
     merged by a second build), the dataset item and collate, a config with a "split", the command line end to end; startable asks for the marker""",
    """  F  window images on nuScenes mini: each cached older image is the JPEG of its frame's sample_data (independent walk along `prev`), the resumable builder and its finished-build markers (key frames and
     older frames: scenes and the builder's constants, merged by a second build under a lock, none after a crash half-way), the dataset item and collate, a config with a "split", the command line end
     to end (also on a student 'all' config with feature caches named: the t26 one); startable asks for both markers""")
rep("""  E  image jitter: known answers against an independent numpy reference, one draw per sample for all its frames, deterministic in (seed, step, micro-batch), a function of the step and not of a random
     stream, the key frames of a clip share the draw, off below img_jitter_from, none in evaluation mode, the anchor target stays the clean image's""",
    """  E  image jitter: known answers against an independent numpy reference, one draw per sample for all its frames, deterministic in (seed, step, micro-batch), a function of the step and not of a random
     stream, the key frames of a clip share the draw, off below img_jitter_from, none in evaluation mode, the anchor target stays the clean image's, the older frames' noise is its own stream (in the
     model), and the real train() (toy network) hands run_clip the step and the micro-batch""")
open(p, "w").write(s)
