#ifndef HALO_EXCHANGE_H
#define HALO_EXCHANGE_H
#include "mpi_setup.h"


//Halo exchange on the finest level.
//Phi has dimension [2*(width_x+2)*(width_t+2)]
void exchange_halo(c_double* phi);


#endif