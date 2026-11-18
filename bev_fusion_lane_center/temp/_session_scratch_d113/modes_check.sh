#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 bash tools/env/_py14.sh token_readout.py modes 0 e374_m_s0 e374_z_s0 0.1,0.6,off > temp/_session_scratch_d113/modes_check.out 2>&1
