cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" GAL_SNAP=1 GAL_OUT=errors_snap_e312
mkdir -p temp/jr/media/errors_snap_e312
nice -n 10 bash tools/env/_py14.sh ../jr/fail_gallery.py e312_p_s0 0 0.1 > temp/jr/gallery_e312_run.log 2>&1
echo GALLERY_E312_DONE >> temp/jr/gallery_e312_run.log
