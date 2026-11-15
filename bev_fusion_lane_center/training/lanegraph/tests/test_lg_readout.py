"""Unit tests of the hard output steps on toy inputs (no data, no model, no project code)."""
import dataclasses
import math

import numpy as np

from lg_config import CHAMPION
from lg_graph import simplify
from lg_readout import chains, deliver, suppress_close_nodes


def test_suppression():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [5.0, 0.0], [5.2, 0.0]])
    e = np.array([0.9, 0.95, 0.6, 0.4, 0.8])
    keep = suppress_close_nodes(pos, e, 1.5, 0.5)
    assert keep.tolist() == [False, True, False, False, True], keep          # nodes 0 and 2 lie within 1.5 m of the stronger node 1; node 3 is below the threshold
    assert suppress_close_nodes(np.array([[0.0, 0.0], [1.5, 0.0]]), np.array([0.9, 0.8]), 1.5, 0.5).tolist() == [True, True]   # exactly at the radius both stay


def test_chains():
    assert chains([(0, 1), (1, 2), (2, 3), (2, 4)]) == [[0, 1, 2], [2, 3], [2, 4]]
    assert chains([(5, 6), (6, 5)]) == [[5, 6, 5]]                             # a pure cycle is one closed chain


def test_simplify():
    P = np.c_[np.linspace(0, 10, 21), np.zeros(21)]
    assert len(simplify(P, 0.3)) == 2
    L = np.r_[np.c_[np.linspace(0, 5, 11), np.zeros(11)], np.c_[np.full(10, 5.0), np.linspace(0.5, 5, 10)]]     # an L: one corner
    assert np.allclose(simplify(L, 0.3), [[0, 0], [5, 0], [5, 5]])


def toy(long_link: bool = False, allowed: bool = True, entry_turn: float = 0.0):
    """a lane before (nodes 0-2, heading +y) and after a junction (nodes 3-5); optionally a long link 2 -> 3 of B; one proposed connector whose
    entry keypoint lies 1 m beside the end of the first lane (its heading turned by entry_turn degrees) and whose exit lies 0.5 m beside the second"""
    pos = np.array([[0.0, y] for y in (-12.0, -10.0, -8.0)] + [[8.0, y] for y in (8.0, 10.0, 12.0)])
    src, dst = [0, 1, 3, 4] + ([2] if long_link else []), [1, 2, 4, 5] + ([3] if long_link else [])
    n = len(src)
    lanes = dict(pos=pos, dir=np.tile([0.0, 1.0], (6, 1)), e=np.full(6, 0.9), valid=np.ones(6, bool), start=np.array([0, 0, 0, 3, 3, 3]),
                 src=np.array(src), dst=np.array(dst), p=np.full(n, 0.9), sa=np.full(n, 1 / 3), sb=np.full(n, 1 / 3), long=np.arange(n) == 4)
    kp = np.zeros((4, 2), np.float32)
    kp[0], kp[2] = [1.0, -8.0], [8.5, 8.0]
    kd = np.tile([0.0, 1.0], (4, 1)).astype(np.float32)
    kd[0] = [math.sin(math.radians(entry_turn)), math.cos(math.radians(entry_turn))]
    pairs = dict(kp_pos=kp, kp_dir=kd, logit=np.array([[5.0, -9], [-9, -9]], np.float32), allowed=np.array([[allowed, True], [True, True]]),
                 scale=np.full((2, 2, 2), 1 / 3, np.float32), att_pos=kp + np.float32(0.25), att_dir=kd)
    return lanes, pairs


def test_snap():
    g = deliver(*toy(), CHAMPION)
    assert g.kinds == ["lane", "lane", "connector"] and sorted(g.edges) == [(0, 2), (2, 1)], (g.kinds, g.edges)
    assert np.allclose(g.lines[2][0], [0.0, -8.0]) and np.allclose(g.lines[2][-1], [8.0, 8.0])
    assert g.nodes[2] == [10000, 20000]


def test_angle_gate():
    lanes, pairs = toy(entry_turn=42.0)
    assert (0, 2) in deliver(lanes, pairs, CHAMPION).edges                                       # 42 degrees: inside the 45 degree gate
    g = deliver(lanes, pairs, dataclasses.replace(CHAMPION, wire_angle=40.0))
    assert (0, 2) not in g.edges and (2, 1) in g.edges and np.allclose(g.lines[2][0], [1.0, -8.0])   # outside: not wired, the curve keeps its keypoint


def test_cover():
    g = deliver(*toy(long_link=True), CHAMPION)
    assert g.kinds == ["lane", "lane", "connector"], g.kinds                                     # B's long link 2 -> 3 is covered by the connector: dropped
    g = deliver(*toy(long_link=True, allowed=False), CHAMPION)
    assert g.kinds == ["lane", "lane", "link"] and sorted(g.edges) == [(0, 2), (2, 1)], (g.kinds, g.edges)   # no connector: B's long link stays, wired


def test_allowed():
    assert deliver(*toy(allowed=False), CHAMPION).kinds == ["lane", "lane"]                     # a pair that is not allowed gives no connector, whatever its logit


def test_attached_ends():
    g = deliver(*toy(), dataclasses.replace(CHAMPION, attached_ends=True))
    assert np.allclose(g.lines[2][0], [1.25, -7.75]) and np.allclose(g.lines[2][-1], [8.75, 8.25])   # the curve runs between the attached ends ...
    assert sorted(g.edges) == [(0, 2), (2, 1)]                                                   # ... and the proximity rule wires it


TESTS = [test_suppression, test_chains, test_simplify, test_snap, test_angle_gate, test_cover, test_allowed, test_attached_ends]
