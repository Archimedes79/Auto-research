cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
until grep -q MANEUVER_DONE temp/jr/maneuver_run.log 2>/dev/null && [ "$(tail -n 1 temp/jr/maneuver_run.log)" = "MANEUVER_DONE" ]; do sleep 10; done
bash tools/env/_py14.sh ../jr/fail_gallery.py e308_t_s0 0 0.5 > temp/jr/gallery_run.log 2>&1
echo GALLERY_DONE >> temp/jr/gallery_run.log
