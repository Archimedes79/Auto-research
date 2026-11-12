import inspect, torch
print(torch.__version__)
print(inspect.getsource(torch.cuda.get_rng_state)[:900])
print(inspect.getsource(torch.cuda.random.get_rng_state_all)[:400])
