export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
export MPI_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include
export MPI_CXX_LIBRARIES=/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
export CFLAGS="-I$MPI_INCLUDE_PATH"
export CPPFLAGS="-I$MPI_INCLUDE_PATH"
export LDFLAGS="-L$MPI_CXX_LIBRARIES -lmpi"
export HDF5_MPI="ON"
pip install --no-cache-dir --no-binary=h5py h5py  > .log.txt 2>&1 & code .log.txt
