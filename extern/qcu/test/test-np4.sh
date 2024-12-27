bash ./make.sh
rm log_*
mpirun -np 4 ./test
