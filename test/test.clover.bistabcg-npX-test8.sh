pushd ../
# source ./env.sh
# bash ./install.sh
popd
rm _*
mpirun -x UCX_TLS=self,sm,rc -np 64 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test8.py > test.clover.bistabcg-np64-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 128 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test8.py > test.clover.bistabcg-np128-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 256 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test8.py > test.clover.bistabcg-np256-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 512 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test8.py > test.clover.bistabcg-np512-test8.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 1024 "/public/home/zhangxin80699/openmpi_bind_mlnx.sh" python -u ./test.clover.bistabcg-test8.py > test.clover.bistabcg-np1024-test8.log 2>&1