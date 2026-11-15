p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep("(card t21, Mirko's D51); CPU, seconds, no data.", "(card t21, Mirko's D51); CPU, about 2 min, no data.")
rep('''  F  mutants: eight planted bugs''', '''  H  the trainer's own loop: `train.run_clip` over two key frames hands the first one's buffer to the second (recurrent), never in a pair batch;
     the gradients of the lift and the temporal block exist.
  F  mutants: eight planted bugs''')
open(p, "w", encoding="utf-8", newline="").write(s)
