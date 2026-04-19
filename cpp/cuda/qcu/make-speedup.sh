# init
echo "There is init!"
# tmpfs
mount -t tmpfs tmpfs . -o size=100M
# source
source ./env.sh
# make
ln -s CMakeLists-nv.txt CMakeLists.txt
cmake .
# apt install ccache ## accelerating make...
ccache make -j$(nproc)
# clean
rm -rf CMakeFiles
rm cmake_install.cmake
rm CMakeCache.txt
rm Makefile