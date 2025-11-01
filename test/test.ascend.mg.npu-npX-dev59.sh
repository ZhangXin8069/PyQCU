pushd ../
# source ./env.sh
# bash ./install.sh
popd
rm _*
mpirun -np 1 python -u ./test.ascend.mg.npu-dev59.py > test.ascend.mg.npu-np1-dev59.log 2>&1
mpirun -np 2 python -u ./test.ascend.mg.npu-dev59.py > test.ascend.mg.npu-np2-dev59.log 2>&1
mpirun -np 4 python -u ./test.ascend.mg.npu-dev59.py > test.ascend.mg.npu-np4-dev59.log 2>&1
mpirun -np 8 python -u ./test.ascend.mg.npu-dev59.py > test.ascend.mg.npu-np8-dev59.log 2>&1