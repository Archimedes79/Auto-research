p = "temp/_session_scratch_s292/new/test_mnms_layer.py"
s = open(p, encoding="utf-8").read()
def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
sub("# ---- 5. the loss sees the damped existence e' (not the head's own e_raw): with the layer on, the node loss differs from the plain model's on the same batch\nl_plain = grad_sums(plain, 3)[2]\nl_mm = grad_sums(mm, 3)[2]\n",
    "# ---- 5. the loss sees the damped existence e' (not the head's own e_raw): with the layer on, the node loss differs from the plain model's on the same batch. At a fresh init the offsets are about 0 and\n"
    "# cells 2 m apart do not overlap (the layer does nothing), so the node head is wrapped: its odd columns move 1.5 m toward the left neighbour (every pair of neighbours overlaps)\n"
    "import copy  # noqa: E402\n\n\n"
    "class Squeeze(torch.nn.Module):\n"
    "    def __init__(self, head):\n"
    "        super().__init__()\n"
    "        self.head = head\n\n"
    "    def forward(self, *a):\n"
    "        o = dict(self.head(*a))\n"
    "        pos = o[\"pos\"].clone()\n"
    "        pos[:, 1, :, 1::2] = pos[:, 1, :, 1::2] - 1.5\n"
    "        o[\"pos\"] = pos\n"
    "        return o\n\n\n"
    "def squeezed(model):\n"
    "    model = copy.deepcopy(model)\n"
    "    model.nodes = Squeeze(model.nodes)\n"
    "    return model\n\n\n"
    "plain_s, mm_s = squeezed(plain), squeezed(mm)\n"
    "l_plain = grad_sums(plain_s, 3)[2]\n"
    "l_mm = grad_sums(mm_s, 3)[2]\n")
sub("    l_mut = grad_sums(mm, 3)[2]", "    l_mut = grad_sums(mm_s, 3)[2]")
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
