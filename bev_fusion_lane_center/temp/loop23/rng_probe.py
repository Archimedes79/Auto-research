import torch
st = torch.get_rng_state().cuda()
try:
    torch.set_rng_state(st)
    print("old code would pass")
except Exception as e:
    print("old code raises:", type(e).__name__, str(e)[:80])
