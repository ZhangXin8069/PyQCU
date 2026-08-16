#!/bin/bash

N=512   #Number of lattice sites on x or t
SAP_BLOCK_NUMBER=4 #Number of SAP blocks on the x and t direction on a local rank
BSIZE=8   #Blocks size
NBLOCKS=$((N/BSIZE)) #Number of blocks across the whole lattice
NV=10   #Number of test vectors
RANKS_X=4 #MPI ranks on the x direction
RANKS_T=4 #MPI ranks on the t direction
LEVELS=3 #Number of levels
M0=-0.1023 #Bare mass parameters
CONFPATH="2D_U1_Ns512_Nt512_b40000_m-01023_0.ctxt"
RHSPATH="rhs_conf0_512x512.rhs"
PARAMETERS_PATH="parameters"
CMAKELISTS="CMakeLists.txt"
COMPILE=1 #1 Compile code, anything different doesn't compile 

#Two levels
if [ $LEVELS -eq 2 ]; then
    printf "%d " 0 > parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NV} >> parameters
    printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
    printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters
elif [ $LEVELS -eq 3 ]; then
    printf "%d " 0 > parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NV} >> parameters
    printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
    printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters

    printf "%d " 1 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 10 >> parameters
    printf "%d " 2 >> parameters
    printf "%d\n" 2 >> parameters
elif [ $LEVELS -eq 4 ]; then
    printf "%d " 0 > parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NBLOCKS} >> parameters
    printf "%d " ${NV} >> parameters
    printf "%d " ${SAP_BLOCK_NUMBER} >> parameters
    printf "%d\n" ${SAP_BLOCK_NUMBER} >> parameters

    printf "%d " 1 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 16 >> parameters
    printf "%d " 10 >> parameters
    printf "%d " 2 >> parameters
    printf "%d\n" 2 >> parameters

    printf "%d " 2 >> parameters
    printf "%d " 8 >> parameters
    printf "%d " 8 >> parameters
    printf "%d " 10 >> parameters
    printf "%d " 1 >> parameters
    printf "%d\n" 1 >> parameters
fi

#Inputs
printf "%d\n" ${RANKS_X} > inputs
printf "%d\n" ${RANKS_T} >> inputs
printf "%d\n" ${LEVELS} >> inputs
printf "%f\n" ${M0} >> inputs
printf "%s\n" ${CONFPATH} >> inputs
printf "%s\n" ${RHSPATH} >> inputs
printf "%s\n" ${PARAMETERS_PATH} >> inputs 

if [ $COMPILE -eq 1 ]; then
    sed -i "23s/set(NS \".*\")/set(NS \"${N}\")/" "$CMAKELISTS"
    sed -i "24s/set(NT \".*\")/set(NT \"${N}\")/" "$CMAKELISTS"
    rm -rf build
    mkdir build
    cd build
    cmake ../
    make
    mv DDAlpha_${N}x${N} ../
    cd ../
fi

mpirun --oversubscribe -n 16 DDAlpha_${N}x${N} < inputs
