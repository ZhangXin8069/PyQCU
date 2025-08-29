pushd ../
# source ./env.sh
# bash ./install.sh
popd
# mpirun -x UCX_TLS=self,sm,rc -np 32 python -u ./test.gauge-test8.py >test.gauge-np32-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 256 python -u ./test.gauge-test8.py >test.gauge-np256-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 32 python -u ./test.clover.bistabcg-test8.py >test.clover.bistabcg-np32-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 64 python -u ./test.clover.bistabcg-test8.py >test.clover.bistabcg-np64-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 128 python -u ./test.clover.bistabcg-test8.py >test.clover.bistabcg-np128-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 256 python -u ./test.clover.bistabcg-test8.py >test.clover.bistabcg-np256-test8.log 2>&1
