# BUGFIX 2026-07-28 R3: add set -e and && chaining for error detection.
set -e
# init
echo "There is init!"
# source
source ./env.sh
# make (with error detection via && chaining)
ln -sf CMakeLists-nv.txt CMakeLists.txt
cmake . && make -j$(nproc)
# clean (tolerate missing files since cmake may have failed before creating them)
rm -rf CMakeFiles
rm -f cmake_install.cmake CMakeCache.txt Makefile
echo "make.sh: SUCCESS"