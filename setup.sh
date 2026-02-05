mkdir -p ./lib
pushd ./src/cuda/qcu
bash ./make.sh
mv ./libqcu.so ../../../lib
popd
