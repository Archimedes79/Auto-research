import sys, os
sys.path.insert(0, sys.argv[1])
import decide as D, experiment as X
def fake(cards):
    X.arm_cards = lambda exp, a, root=None: cards[a]
exp = dict(id="E1", metric="Q", direction="higher", band=0.01, control="a", arms={"a": {}, "b": {}},
           rule=[dict(metric="Q", max_loss=0.005)], prediction=dict(value=0.0, vs="a"))
# one seed each, variant +0.05 better
fake({"a": {"": [({"Q": 0.70}, None, "a0")]}, "b": {"": [({"Q": 0.75}, None, "b0")]}})
print("headline max_loss, 1 seed, +0.05:", D.decide_experiment(exp, ".")["verdict"])
# two seeds disagreeing
fake({"a": {"": [({"Q": 0.70}, None, "a0"), ({"Q": 0.70}, None, "a1")]}, "b": {"": [({"Q": 0.80}, None, "b0"), ({"Q": 0.69}, None, "b1")]}})
print("headline max_loss, seeds disagree, mean +0.045:", D.decide_experiment(exp, ".")["verdict"])
print("typed interface same case:", D.decide([0.80, 0.69], [0.70, 0.70])[0])
