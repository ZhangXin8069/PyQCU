#ifndef IO_H
#define IO_H

#include "mpi_setup.h"
#include <string>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <chrono>

//Function for reading configurations or rhs
void read_binary(const std::string& name,const spinor& U);

void broadcast_file_name(std::string& name);

void writeMetadata(int cycle,double tol);

#endif