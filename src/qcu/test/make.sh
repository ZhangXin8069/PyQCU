# source ./env.sh
cmake .
make -j16
# mv libqcu.so ./lib
bash ./clean.sh