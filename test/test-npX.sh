#!/bin/bash
command=$@
pushd ../
source ./env.sh
# bash ./install.sh
popd
# mpirun --allow-run-as-root -n ${command} python ./test.io.py
# mpirun --allow-run-as-root -n ${command} python ./test.set.py
# mpirun --allow-run-as-root -n ${command} python ./test.laplacian.py
# mpirun --allow-run-as-root -n ${command} python ./test.wilson.cg.py
# mpirun --allow-run-as-root -n ${command} python ./test.wilson.bistacg.py
mpirun --allow-run-as-root -n ${command} python ./test.wilson.dslash.py
# mpirun --allow-run-as-root -n ${command} python ./test.clover.dslash.py
# mpirun --allow-run-as-root -n ${command} python ./test.wilson.bistabcg.dslash.eigen.py
