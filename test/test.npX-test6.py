#!/bin/bash
command=$@
pushd ../
source ./env.sh
popd
mpirun -n ${command} python ./test.wilson.bistacg-test6.py