pushd ../
source ./env.sh
bash ./install.sh
popd
mpirun --allow-run-as-root -n 1 python ./test.wilson.bistacg.py 
# mpirun --allow-run-as-root -n 1 python ./test.wilson.dslash.py 
