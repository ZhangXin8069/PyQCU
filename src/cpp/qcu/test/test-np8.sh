bash ./make.sh
rm log_*
mpirun -np 8 ./test
