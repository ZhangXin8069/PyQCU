pushd ../
# source ./env.sh
bash ./install.sh
popd
rm _*
mpirun -x UCX_TLS=self,sm,rc -np 1 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test6.py > test.clover.bistabcg-np1-test6.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 2 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test6.py > test.clover.bistabcg-np2-test6.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 4 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test6.py > test.clover.bistabcg-np4-test6.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 8 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test6.py > test.clover.bistabcg-np8-test6.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 16 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test6.py > test.clover.bistabcg-np16-test6.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 32 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test6.py > test.clover.bistabcg-np32-test6.log 2>&1