# nuScenes - raw data shared by all problems

This folder (`datasets/NuScenes/`: the README and the tools are in git as recipes, the data itself is not: it is on D:) describes the raw nuScenes data and holds the download / copy / check tools. Licence: nuScenes is for non-commercial use; downloads only with Mirko's OK.

## Rule (Mirko, 2026-10-25)
- RAW data only on D:, in `D:\Auto-research\NuScenes` (WSL: `/mnt/d/Auto-research/NuScenes`): exactly what the nuScenes archives contain, plus the archives themselves. Nothing generated is written there.
- DERIVED data (info files, filtered radar, GT, caches) belongs to the problem that makes it, in its project folder on C: (e.g. `bev_fusion_lane_center/data/nuscenes_additions/`, later `bev_fusion/data/`), listed in that project's `data/README.md`.
- Existing code reads `~/data/nuscenes_full` and `~/data/nuscenes` (mini). These are WSL "views": folders of symlinks only, raw entries -> D:, the lane project's additions -> its project folder. Built by `tools/link_views.sh`; re-run it after the repository folder or the D: root moves (the links are absolute).

State 2026-10-25 17:06: raw data copied to D: and verified (787,847 + 27,300 files by size against the manifests, 400 random md5 checks); views built; smoke test (`tools/history/smoke_test_views.py`: devkit on mini, a keyframe image, a radar sweep, the map expansion, a CRN info file) passes. The former WSL copies (about 32 GB) were purged with Mirko's OK (2026-10-25); C: gets the space back after a compaction of the WSL disk image (`../compact_wsl_disk.ps1`, admin, only when nothing runs in WSL).

## Layout on D: (`D:\Auto-research\NuScenes`)
| folder | content |
|---|---|
| `nuscenes_full/` | v1.0-trainval: `v1.0-trainval/` (metadata, 850 scenes), `maps/` (incl. expansion v1.3, 4 cities), `samples/` (keyframes: 6 cameras, 5 radars, parts 1-6), `sweeps/` (radar sweeps parts 1-6; camera sweeps parts 1-6 being unpacked), `LICENSE` |
| `nuscenes_mini/` | v1.0-mini (environment tests) |
| `downloads/` | archives: `nuscenes_camera/pNN/` (camera parts), `nuscenes_radar_456/`; one log per part |
| `manifests/` | file lists (path, size) of the copy from WSL, used to check it |
| `_to_delete/` | scratch of tests (empty folders / small files) |

## What is on disk (2026-10-25)
| sensor data | parts | status |
|---|---|---|
| metadata, maps | all | on D: (copied from WSL 2026-10-25) |
| camera keyframes, six cameras (2 Hz) | 1-6 | on D: (20,418 frames per camera) |
| camera sweeps, six cameras (12 Hz) | 1-6 | on D: since 2026-10-25 21:07 (97.9 GB, approved by Mirko; each archive checked against its S3 ETag; only `sweeps/` unpacked). Complete (`tools/history/check_camera_sweeps.py`): every camera sweep the metadata lists for the 510 scenes is present, 584,074 images (CAM_FRONT 98,029), none missing, none from other scenes. Archives kept in `downloads/nuscenes_camera/pNN/` |
| radar samples + sweeps, five radars | 1-6 | on D: |
| lidar (LIDAR_TOP) | - | not downloaded (parts 1-6: 80.2 GB) |
| parts 7-10 (340 scenes) | - | not downloaded (camera 79.4 GB, radar 0.8 GB, lidar 54.8 GB) |

Archive sizes and ETags: `wsl python3 datasets/NuScenes/tools/list_nuscenes_bucket.py trainval` (listing only).

## Tools (`tools/`)
| tool | what it does |
|---|---|
| `list_nuscenes_bucket.py [filter]` | lists the archives of the public bucket `motional-nuscenes` with size and ETag (no download) |
| `get_blob_part.sh <part> <camera\|lidar\|radar>` | downloads one part (resumes a partial archive), checks the S3 ETag, unpacks it with `extract_blob.sh`, marks it `.unpacked`; environment `NUSC_DL` (download root) and those of `extract_blob.sh` |
| `get_parts_parallel.sh <kind> <part>...` | several parts in parallel (one connection gets 1-2 MB/s, six together about 6 MB/s) |
| `extract_blob.sh <tgz> [--delete]` | path check (no absolute or `..` paths), then unpack; `NUSC_DEST` target, `NUSC_MEMBERS` (e.g. `sweeps`), `NUSC_WINTAR=1` unpacks with Windows `tar.exe` (fast on D:; WSL writes small files there at only about 50 per second) |
| `verify_s3_etag.py <file> <etag>` | checks a file against a plain or multipart S3 ETag |
| `link_views.sh` | builds the WSL views (above); moves the former WSL copies to `~/_to_delete/` |
| `stop_downloads.sh` | stops running downloads (partial archives are kept) |
| `move_additions.sh` | one-off of 2026-10-25: first separation of the lane project's additions inside WSL (superseded by `link_views.sh`) |
| `history/` | one-off scripts of 2026-10-25 (first download run, D: speed test, copy to D: / C: with manifests, `verify_copy.ps1`, `sample_checksums.sh`, self-test of `extract_blob.sh`) |

## Speeds measured from WSL on D: (2026-10-25)
1 GB sequential write 4.2 s, read 4.5 s; 500 files of 200 KB: write 9.5 s, read 1.5 s - enough for images at training rates; fast caches stay in the project folders.
