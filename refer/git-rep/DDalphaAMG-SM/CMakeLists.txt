cmake_minimum_required(VERSION 3.26.5)
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED True)
project(DDalphaAMG VERSION 1.0)


set(SOURCES
    src/variables.cpp           #Relevant parameters for the method, the lattice, MPI, etc.
    src/utils.cpp               #Utility functions
    src/halo_exchange.cpp       #Implementation of the halo exchange for a field on the finest level
    src/dirac_operator.cpp      #Implementation of D, D^+ on the finest level
    src/conjugate_gradient.cpp  #Conjugate gradient and Bi-cgstab (to compare solutions)
    src/fgmres.cpp              #FGMRES class
    src/sap.cpp                 #SAP class
    src/level.cpp               #Multigrid level class
    src/tests.cpp               #Different tests, to be removed in the future.
    src/amg.cpp                 #Algebraic multigrid
    src/methods.cpp             #Class for comparing different matrix-inversion methods
    src/io.cpp                  #For reading/writing configurations or rhs
    src/main.cpp                #Main program
)

set(NS "512")   #Lattice dimension x
set(NT "512")   #Lattice dimension t

configure_file(${CMAKE_SOURCE_DIR}/include/config.h.in ${CMAKE_SOURCE_DIR}/include/config.h)


add_executable("DDAlpha_${NS}x${NT}" ${SOURCES})


target_include_directories("DDAlpha_${NS}x${NT}"
        PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
)
target_compile_options("DDAlpha_${NS}x${NT}" PRIVATE
			-O3
            #-g #For profiling
)


find_package(MPI REQUIRED)
target_link_libraries("DDAlpha_${NS}x${NT}" PUBLIC MPI::MPI_CXX)


#PROJECT_SOURCE_DIR: it indicates where src is located
#cmake -G "MinGW Makefiles" -DCMAKE_CXX_COMPILER="C:/msys64/ucrt64/bin/g++.exe" -DCMAKE_C_COMPILER="C:/msys64/ucrt64/bin/gcc.exe" ../
#dir/s/b *.txt > confs.txt