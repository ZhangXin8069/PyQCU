# init
echo "There is init!"
# source
source ./env.sh
# make
cmake .
make -j$(nproc)
# clean
rm -rf CMakeFiles
rm cmake_install.cmake
rm CMakeCache.txt
rm Makefile
rm -rf build
# export
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
