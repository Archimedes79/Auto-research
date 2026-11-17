import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep('''The key-weight switch of the head counts''', '''Architecture options (all in the recipe and in `model.pt`, `load_model` rebuilds the same head through `build_head`; the defaults are the head of the first build, bit for bit): `--d` 64, `--layers` 3, `--heads` 4, `--ffn` (default 2 x d).
`--token-drop P` removes each valid token with probability P in every training step (a new draw per step, seeded by (seed, the head's step counter, the frame token)): the removed token is not valid, has type none and loses every target pair (`drop_tokens`,
the same effect on the batch as the loader's own `token_drop`, which draws once per frame and cache seed, the same removal every epoch); the targets are NOT re-contracted over the removed tokens (GT links through a removed token are lost, not bridged).
The key-weight switch of the head counts''')
open(p, "w", encoding="utf-8").write(s)
