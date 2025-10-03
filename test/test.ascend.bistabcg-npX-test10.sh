pushd ../
# source ./env.sh
# bash ./install.sh
popd
rm _*
mpirun -x UCX_TLS=self,sm,rc -np 1 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np1-test10.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 2 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np2-test10.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 4 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np4-test10.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 8 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np8-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 16 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np16-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 32 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np32-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 64 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np64-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 128 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np128-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 256 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np256-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 512 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np512-test10.log 2>&1
# mpirun -x UCX_TLS=self,sm,rc -np 1024 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.bistabcg-test10.py > test.ascend.bistabcg-np1024-test10.log 2>&1