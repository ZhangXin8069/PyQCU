mkdir -p ./lib
pushd ./extern/qcu
bash ./make.sh
mv ./libqcu.so ../../lib
popd