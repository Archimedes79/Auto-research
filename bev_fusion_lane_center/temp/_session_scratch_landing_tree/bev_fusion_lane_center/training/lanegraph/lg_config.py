"""Every selection and read-out setting of the champion (system of record E324 s, decision D94) in one frozen dataclass.
Not here: the shapes of the trained layers and the fixed scalings of their input features (e.g. the / 20 m of a pair feature), which are part of the weights."""
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]          # bev_fusion_lane_center/


@dataclass(frozen=True)
class Config:
    # B, the lane head (E273): candidate sets (index choices, no gradient through the choice)
    max_nodes: int = 512              # active nodes per frame (top-K existence) ...
    node_min: float = 0.05            # ... with existence above this
    short_cells: int = 4              # a short link joins nodes at most this many 2 m cells apart in each axis ...
    short_behind: float = -1.0        # ... whose offset along the source heading is above this (m)
    short_reach: float = 8.0          # a link longer than this (m) is a long link (learned handle scales)
    long_ends: int = 64               # long links: the strongest long-link starts x the strongest ends ...
    long_max: float = 100.0           # ... at most this far apart (m) ...
    long_cone: float = 75.0           # ... inside this forward cone (degrees)
    link_samples: int = 16            # points of a link curve where the centre-lane evidence is read
    # the junction head (E307 maps + E324 s pair head + E320 attachment)
    keypoints: int = 32               # entry and exit keypoints per frame, each: 3 x 3 heat peaks ...
    peak_min: float = 0.1             # ... above this
    turn_max: float = 130.0           # an allowed pair turns by at most this (degrees) ...
    chord_min: float = 2.0            # ... over a chord of at least this (m)
    evidence_samples: int = 12        # points of the pair curve where the evidence maps are read
    group_angle: float = 25.0         # order rule: lane ends form a group within this heading difference (degrees) ...
    group_along: float = 6.0          # ... and these distances along / across (m)
    group_across: float = 6.0
    attach_nodes: int = 8             # attachment: the nearest nodes of B ...
    attach_reach: float = 4.0         # ... within this distance (m; also the scale of its distance features) ...
    attach_angle: float = 45.0        # ... and heading difference (degrees) ...
    attach_node_min: float = 0.05     # ... whose existence is above this
    # the named hard output steps (read-out)
    node_thr: float = 0.5             # a delivered node: existence >= node_thr ...
    close_radius: float = 1.5         # ... and no stronger delivered node closer than this (E275)
    link_thr: float = 0.5             # a delivered link of B
    pair_thr: float = 0.1             # a delivered connector: pair probability >= pair_thr (allowed pairs only)
    wire_radius: float = 4.0          # a connector end is wired to the nearest piece end within this (m) ...
    wire_angle: float = 45.0          # ... and this heading difference (degrees)
    cover_radius: float = 4.0         # a long link of B whose two ends a connector covers is dropped
    attached_ends: bool = False       # False: the card of record (connector between the wired piece ends, E311);
                                      # True: connector between the learned attached ends, wired by the same proximity rule (E320 / E324 'primary')
    lane_step: float = 0.5            # sampling of B's link curves (m)
    connector_step: float = 0.25      # sampling of a connector: max(connector_min_points, int(connector_density x chord / connector_step)) points
    connector_density: float = 1.6
    connector_min_points: int = 8
    connector_ids: tuple = (10000, 20000)   # node ids of connector n: (entry id + n, exit id + n)
    dp_tol: float = 0.3               # Douglas-Peucker tolerance of the written polylines (m)

    @property
    def cos_cone(self) -> float:
        return math.cos(math.radians(self.long_cone))

    @property
    def cos_turn(self) -> float:
        return math.cos(math.radians(self.turn_max))


CHAMPION = Config()


def run_paths(seed: int) -> tuple:
    """the two checkpoints of the champion of one seed: B (pixels, nodes, links) and the junction system (maps, pair head, attachment)"""
    return PROJECT / "runs" / f"e273_p_s{seed}" / "model.pt", PROJECT / "runs" / f"e324_s_s{seed}" / "model.pt"
