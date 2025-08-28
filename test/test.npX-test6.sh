#!/bin/bash
command=$@
pushd ../
source ./env.sh
popd
mpirun -x UCX_TLS=self,sm,rc -n ${command} python -u ./test.wilson.bistacg-test6.py