# datasets/nuscenes - the nuScenes dataset

The raw data all problems share, named by a manifest, with the tools that download, check and link it. In git: this page, `manifest.json` and `tools/`; the data never. nuScenes is for non-commercial
use; downloads only with the owner's OK.

## Where it lives
| what | where (this PC) |
|---|---|
| raw data, unpacked: exactly what the archives contain; nothing generated is written there, never deleted | `D:\Auto-research\NuScenes` (WSL `/mnt/d/Auto-research/NuScenes`) |
| derived data (views, ground truth in `groundtruth/`, a problem's own data in `<problem>/`) | the data root: `data` of the machine file (`research machine`), `D:\Auto-research\data` |
| the views code reads: `<data root>/nuscenes` (trainval) and `<data root>/nuscenes_mini`; older code uses the links `~/data/nuscenes_full` and `~/data/nuscenes` to them | made by `tools/link_views.sh` |
| archives, only while a download is unpacked | `D:\Auto-research\NuScenes_zips` (`NUSC_DL`) |

## Raw data
| folder | content |
|---|---|
| `nuscenes_full/` | v1.0-trainval (850 scenes): `v1.0-trainval/` tables, `maps/` (expansion v1.3), `samples/` and `sweeps/`: radar (five) in parts 1 - 10; camera in parts 1 - 6 (six cameras), parts 7 - 10 CAM_FRONT only; no lidar |
| `nuscenes_mini/` | v1.0-mini (10 scenes) |
| `occ3d/` | Occ3D-nuScenes labels (third party, from nuScenes lidar; MIT + nuScenes non-commercial terms): `gts/<scene>/<sample token>/labels.npz`, `annotations.json` |
| `manifests/` | file lists (path, size) of the raw data and the download logs |

Are the files the ground-truth generator reads all there: `python -m GtGenerator data ~/data/nuscenes_full [--set small|full]`.

## Manifest (`manifest.json`)
It names the raw data by what cannot change unnoticed (hashes of the tables, maps and file lists, a fingerprint per sensor folder, the Occ3D files) and keeps the identity of every downloaded archive,
also after it was deleted. Its `data_identity` hashes the data part only: a ground-truth key names the data by it. Written and compared with the disk by `tools/make_manifest.py` (its `--help` says how).
Regenerate and commit it after any change of the data.

## Archives (D42)
An archive is unpacked completely, compared with the data (`tools/verify_archive.ps1`: every member there with the same size and bytes) and deleted only on 0 differences. The manifest keeps its identity,
so the data can be traced to the download and the download repeated.

## Tools (`tools/`, from the repository root: `wsl bash datasets/nuscenes/tools/<tool>`, `wsl python3 ...`)
| tool | what it does (how: the header of the script) |
|---|---|
| `link_views.sh [--check]` | builds or checks the views; run it again when a root moves |
| `make_manifest.py`, `scan_tree.ps1` | the manifest, its native folder scanner |
| `list_nuscenes_bucket.py [filter]` | the archives of the public bucket with size and ETag (no download) |
| `get_blob_part.sh <part> <camera\|lidar\|radar>`, `get_parts_parallel.sh <kind> <part>...` | download parts (resumable), check the ETag, unpack, compare, delete the archive; `NUSC_DL` and `NUSC_DEST` required |
| `extract_blob.sh <tgz> [--delete]` | path check, unpack to `NUSC_DEST` (`NUSC_MEMBERS`; `NUSC_WINTAR=1`: Windows tar, fast on D:); `--delete`: the comparison, then the removal |
| `verify_archive.ps1`, `verify_s3_etag.py <file> <etag>` | every archive member exists in the data with the same size and bytes; a file against an S3 ETag |
| `list_gdrive_folder.py`, `gdrive_size.sh`, `get_gdrive_file.sh`, `get_occ3d_labels.sh` | list, size and download public Google Drive files; the recipe of the Occ3D labels |
| `stop_downloads.sh` | stops running downloads (partial archives are kept) |

Reading many small files through WSL from D: is slow: bulk checks of the raw data run natively on Windows (`verify_archive.ps1`, `scan_tree.ps1`).
