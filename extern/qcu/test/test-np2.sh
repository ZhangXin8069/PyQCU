bash ./make.sh
rm log_*
mpirun -np 2 ./test
