"""scratch (critic, read-only review): does first_batches() (E379, uncommitted jtoken_train.py) change the data order of the run that follows it? CPU only, tiny dataset, 2 workers."""
import torch

N, BS = 64, 4


class DS(torch.utils.data.Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return i


def mk():
    return torch.utils.data.DataLoader(DS(), batch_size=BS, shuffle=True, num_workers=2, persistent_workers=True, drop_last=True)


def endless(loader):
    while True:
        for b in loader:
            yield b


def first(loader, n):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        it = iter(loader)
        return [next(it).tolist() for _ in range(n)]


def fit_order(loader, n):
    torch.manual_seed(0)
    it = endless(loader)
    return [next(it).tolist() for _ in range(n)]


if __name__ == "__main__":
    a = fit_order(mk(), 3)
    ld = mk()
    f = first(ld, 3)
    b = fit_order(ld, 3)
    print("no-teacher run, first 3 batches:      ", a)
    print("first_batches (teacher start check):  ", f)
    print("with-teacher run, first 3 batches:    ", b)
    print("check order == no-teacher run order:", f == a, "| with-teacher run order == no-teacher run order:", b == a, "| == check:", b == f)

    # proposed fix: the check reads a SECOND loader (workers 0); the run's loader is untouched
    ld2 = mk()
    chk = first(torch.utils.data.DataLoader(DS(), batch_size=BS, shuffle=True, num_workers=0, drop_last=True), 3)
    c = fit_order(ld2, 3)
    print("fix: check order == no-teacher order:", chk == a, "| run order == no-teacher order:", c == a)
