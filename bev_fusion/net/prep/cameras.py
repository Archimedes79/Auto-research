"""The camera feature sources a model config can name (data, not code): which frozen features a kind reads and from which cache."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraSource:
    """One frozen image encoder whose (cached) outputs feed the camera slot: the feature family fixes the depth bins, the cell
    convention and the cache's feature size; the suffix names its batch fields (prob<suffix>, ctx<suffix>, ida<suffix>) and, after the
    first source, its modules (lift<suffix>, cam_reduce<suffix>: the checkpoint keys of the t10 chain)."""
    features: str      # "dualbev" (stride 16, cell centres, depth bins 1.25 + 0.5 i) | "bevfusion" (stride 8, frustum cells, bins 1.0 + 0.5 i)
    cache: str         # TrainConfig field holding the cache folder
    suffix: str        # "" for the first source, "2" for the second


# camera slot: kind -> its feature sources, one lift + height collapse each; two or more are merged (Conv 1x1 over the concatenated BEVs).
# The first source is the geometry reference (visibility, precompute); the dataset, collate_cached, the cached model, train.build and
# the tests read this table, not the kind's name.
CAMERAS = {"dualbev": (CameraSource("dualbev", "feature_cache", ""),),
           "bevfusion": (CameraSource("bevfusion", "feature_cache", ""),),
           "dual": (CameraSource("dualbev", "feature_cache", ""), CameraSource("bevfusion", "feature_cache2", "2"))}
