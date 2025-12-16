pushd ../
# source ./env.sh
# bash ./install.sh
popd
mpirun --allow-run-as-root -np 2 python -u ./test.laplacian-test5.py > test.laplacian.x99-test5.log 2>&1
mpirun --allow-run-as-root -np 2 python -u ./test.wilson.dslash-test5.py > test.wilson.dslash.x99-test5.log 2>&1
mpirun --allow-run-as-root -np 2 python -u ./test.wilson.cg-test5.py > test.wilson.cg.x99-test5.log 2>&1
mpirun --allow-run-as-root -np 2 python -u ./test.wilson.bistabcg-test5.py > test.wilson.bistabcg.x99-test5.log 2>&1
mpirun --allow-run-as-root -np 2 python -u ./test.clover.dslash-test5.py > test.clover.dslash.x99-test5.log 2>&1