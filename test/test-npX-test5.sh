pushd ../
# source ./env.sh
# bash ./install.sh
popd
mpirun -x UCX_TLS=self,sm,rc -np 2 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.wilson.dslash-test5.py > test.wilson.dslash-test5.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 2 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.wilson.cg-test5.py > test.wilson.cg-test5.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 2 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.wilson.bistabcg-test5.py > test.wilson.bistabcg-test5.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 2 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.clover.dslash-test5.py > test.clover.dslash-test5.log 2>&1