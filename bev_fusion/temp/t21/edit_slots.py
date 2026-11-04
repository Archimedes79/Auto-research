p = "tests/test_slots.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''         ("cam_temporal", "window_memory"), ("cam_temporal", "ego_gru"), ("cam_temporal", "none"),
''', '''         ("cam_temporal", "window_memory"), ("cam_temporal", "ego_gru"), ("cam_temporal", "none"), ("cam_buffer", "recurrent"),
''')
rep('''INVALID = [dict(rad_static_gate=True),''', '''INVALID = [dict(cam_buffer="pair", cam_temporal="none"), dict(cam_buffer="recurrent", cam_temporal="ego_gru"), dict(cam_buffer="stack"),   # t21 (pair: test_stream_temporal.py)
           dict(rad_static_gate=True),''')
open(p, "w", encoding="utf-8", newline="").write(s)
