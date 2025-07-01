mkdir -p ./lib
pushd ./extern/cuda/qcu
bash ./make.sh
mv ./libqcu.so ../../../lib
popd
