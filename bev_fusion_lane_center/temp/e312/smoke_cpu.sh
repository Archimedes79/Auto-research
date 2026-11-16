cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" BFLC_SIG=1.0
nice -n 5 bash tools/env/_py14.sh jends_train.py --name e312_smoke_cpu_s0 --seed 0 --maps $(pwd)/runs/e286_j1_s0/model.pt --e2e --scales --maps-w 1.0 --smoke --workers 1 > temp/e312/smoke_cpu.log 2>&1
echo "exit $?" >> temp/e312/smoke_cpu.log
