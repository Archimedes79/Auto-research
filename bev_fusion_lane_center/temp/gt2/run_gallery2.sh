cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" GAL_SNAP=1 GAL_OUT=errors_snap
until tail -n 1 temp/jr/jman_run.log 2>/dev/null | grep -q JMAN_DONE; do sleep 10; done
mkdir -p temp/jr/media/errors_snap
nice -n 10 bash tools/env/_py14.sh ../jr/fail_gallery.py e305_e_s0 0 0.1 > temp/jr/gallery_snap_run.log 2>&1
echo GALLERY_SNAP_DONE >> temp/jr/gallery_snap_run.log
