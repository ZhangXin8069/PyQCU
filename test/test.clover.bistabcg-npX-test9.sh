pushd ../
# source ./env.sh
# bash ./install.sh
popd
rm _*
mpirun -x UCX_TLS=self,sm,rc -np 128 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test9.py > test.clover.bistabcg-np128-test9.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 256 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test9.py > test.clover.bistabcg-np256-test9.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 512 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test9.py > test.clover.bistabcg-np512-test9.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 1024 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test9.py > test.clover.bistabcg-np1024-test9.log 2>&1