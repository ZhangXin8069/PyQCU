pushd ../
# source ./env.sh
#bash ./install.sh
popd
# mpirun -x UCX_TLS=self,sm,rc -np 1 python -u ./test.wilson.bistabcg-test6.py >test.wilson.bistabcg-np1-test6.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 2 python -u ./test.wilson.bistabcg-test6.py >test.wilson.bistabcg-np2-test6.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 4 python -u ./test.wilson.bistabcg-test6.py >test.wilson.bistabcg-np4-test6.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 8 python -u ./test.wilson.bistabcg-test6.py >test.wilson.bistabcg-np8-test6.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 16 python -u ./test.wilson.bistabcg-test6.py >test.wilson.bistabcg-np16-test6.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 32 python -u ./test.wilson.bistabcg-test6.py >test.wilson.bistabcg-np32-test6.log 2>&1
