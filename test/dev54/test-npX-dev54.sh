command=$@
echo "command:${command}"
pushd ../
# source ./env.sh
# bash ./install.sh
popd
mpirun -x UCX_TLS=self,sm,rc -np ${command} '/public/home/zhangxin80699/openmpi_bind_mlnx.sh' python -u ./test.laplacian.py > .log 2>&1