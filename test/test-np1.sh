pushd ../
bash ./install.sh
popd
mpirun --allow-run-as-root -n 1 python ./test.pyqcu.py
