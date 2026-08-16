#include "halo_exchange.h"


void exchange_halo(c_double* phi){
    using namespace LV; //Lattice parameters namespace
	using namespace mpi;

    int row_size = dof * width_t;
    //Send top row to top rank. Receive top row from bot rank.
    MPI_Sendrecv(&phi[idx(1,1,0)], row_size, MPI_DOUBLE_COMPLEX, top, 0,
        &phi[idx(width_x+1,1,0)], row_size, MPI_DOUBLE_COMPLEX, bot, 0,
        cart_comm, MPI_STATUS_IGNORE);

    //Send bot row to bot rank. Receive bot row from top rank.
    MPI_Sendrecv(&phi[idx(width_x,1,0)], row_size, MPI_DOUBLE_COMPLEX, bot, 1,
        &phi[idx(0,1,0)], row_size, MPI_DOUBLE_COMPLEX, top, 1,
        cart_comm, MPI_STATUS_IGNORE);

    //Send left column to left rank. Receive left column from right rank. 
    MPI_Sendrecv(&phi[idx(1,1,0)], 1, mpi::column_type[0], left, 2,
    &phi[idx(1,width_t+1,0)], 1, mpi::column_type[0], right, 2,
    cart_comm, MPI_STATUS_IGNORE);

    //Send right column to right rank. Receive right column from left rank. 
    MPI_Sendrecv(&phi[idx(1,width_t,0)], 1, mpi::column_type[0], right, 3,
    &phi[idx(1,0,0)], 1, mpi::column_type[0], left, 3,
    cart_comm, MPI_STATUS_IGNORE);

}