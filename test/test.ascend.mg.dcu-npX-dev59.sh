pushd ../
# source ./env.sh
# bash ./install.sh
popd
rm _*
mpirun -x UCX_TLS=self,sm,rc -np 1 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.mg.dcu-dev59.py > test.ascend.mg.dcu-np1-dev59.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 2 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.mg.dcu-dev59.py > test.ascend.mg.dcu-np2-dev59.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 4 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.mg.dcu-dev59.py > test.ascend.mg.dcu-np4-dev59.log 2>&1
mpirun -x UCX_TLS=self,sm,rc -np 8 '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.ascend.mg.dcu-dev59.py > test.ascend.mg.dcu-np8-dev59.log 2>&1