#!/bin/sh
set -e
make clean
sed -i 's/CUDA_ENABLER =.*/CUDA_ENABLER =/g' MakeSettings.mk
make -j -j$(nproc) all
llog ./run -i ./test/test_confs/cpu_4x4_gmres.ini && code .log.txt && rm -rf build dd_alpha_amg  dd_alpha_amg_db &
