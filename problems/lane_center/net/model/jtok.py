"""The pair geometry of the token designs (E357, stage T0 of the token design E355, D105): the features and the candidate gate of every pair of nodes, read by the
champion's head (`jtoken.py`), the token cache and the live tokens; the lattice of the targets.

    with lattice(2.5): ...                                   # `qtargets` on a 2.5 m lattice for the block (N, RES are module constants read at call time; restored after); check_lattice(2.5)
    g, allowed = pair_geometry(pos, dirv, valid)             # (B, N, N, 13) the geometric pair features, (B, N, N) the candidate gate; [b, i, j] = the link i -> j

The allowed gate (a candidate selection by geometry, named in the card): both headings point forward along the chord, turn at most 130 degrees (`COS_TURN`), chord MIN_CHORD .. MAX_CHORD m, both nodes valid, i != j.
E357's own pair head on B's frozen pyramid (JTok, JTokModel, its loss and read-out dump) is in the tag v1. Tests: net/tests/test_jtoken.py (pair_geometry)
"""
import math

import torch

from lane_center.net.prep import qtargets as QT
from lane_center.net.prep.qtargets import check_lattice, lattice  # noqa: E402,F401 (the lattice is a property of the targets; read here by name)

NG = 13                                                                     # geometric pair features
MIN_CHORD, MAX_CHORD = 0.3, 100.0                                           # m: the chord range of the allowed gate
COS_TURN = math.cos(math.radians(130.0))                                    # the largest turn of the allowed gate
SHORT = QT.SHORT                                                            # 8 m: a link with a longer chord is a long link (`qtargets.node_table`)
HALF = QT.HALF


# ------------------------------------------------------------------------------------------------------------------------------ pairs
def pair_geometry(pos, dirv, valid=None, e=None, min_chord=MIN_CHORD, max_chord=MAX_CHORD):
    """pos, dirv (B, N, 2) metres / unit headings, valid (B, N) bool (None = all), e (B, N) existence probability of the nodes (None = 1: GT nodes)
    -> g (B, N, N, 13), allowed (B, N, N) bool; [b, i, j] = the link i -> j. The 13 features (those of the E287 junction pair head): cos and sin of the turn, chord / 20, the chord in the frame of
    heading i and of heading j (along, across; / 20), cos and sin of the chord direction against each heading, e_i, e_j. allowed: both headings forward along the chord, cos(turn) >= cos(130 degrees),
    min_chord <= chord <= max_chord, both valid, i != j."""
    B, N = pos.shape[:2]
    ch = pos[:, None, :, :] - pos[:, :, None, :]
    L = ch.norm(dim=-1)
    ta = dirv[:, :, None, :].expand_as(ch)
    tb = dirv[:, None, :, :].expand_as(ch)
    na = torch.stack([-ta[..., 1], ta[..., 0]], -1)
    nb = torch.stack([-tb[..., 1], tb[..., 0]], -1)
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]  # noqa: E731
    cu = ch / L.clamp_min(1e-6)[..., None]
    cos_turn = dot(ta, tb)
    e = torch.ones_like(pos[..., 0]) if e is None else e
    valid = torch.ones_like(pos[..., 0], dtype=torch.bool) if valid is None else valid
    g = torch.stack([cos_turn, cross(ta, tb), L / 20.0, dot(ch, ta) / 20.0, dot(ch, na) / 20.0, dot(ch, tb) / 20.0, dot(ch, nb) / 20.0, dot(cu, ta), cross(ta, cu), dot(cu, tb), cross(tb, cu),
                     e[:, :, None].expand_as(L), e[:, None, :].expand_as(L)], -1)
    eye = torch.eye(N, dtype=torch.bool, device=pos.device)[None]
    allowed = (cos_turn >= COS_TURN) & (L >= min_chord) & (L <= max_chord) & (dot(ch, ta) > 0) & (dot(ch, tb) > 0) & valid[:, :, None] & valid[:, None, :] & ~eye
    return g, allowed
