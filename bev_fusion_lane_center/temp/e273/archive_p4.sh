#!/bin/bash
cd "$(dirname "$0")/../.."
mkdir -p runs/_superseded
[ -d runs/e273_p_s0_p4 ] && mv runs/e273_p_s0_p4 runs/_superseded/e273_p_s0_v3_p4
ls -d runs/e273_p_s0*
