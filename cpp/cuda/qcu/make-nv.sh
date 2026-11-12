# BUGFIX 2026-07-28 R3: add set -e and && chaining for error detection.
set -e
# init
echo "There is init!"
# source
source ./env.sh
# Compile parallelism follows PYQCU_BUILD_JOBS like the KUPL/DCU entry points;
# nibbling the host CPU count ignores scheduler allocations.
JOBS=${PYQCU_BUILD_JOBS:-$(nproc)}
# make (with error detection via && chaining)
ln -sf CMakeLists-nv.txt CMakeLists.txt
cmake .
make -j"$JOBS"
# clean (tolerate missing files since cmake may have failed before creating them)
rm -rf CMakeFiles
rm -f cmake_install.cmake CMakeCache.txt Makefile
echo "make.sh: SUCCESS"
