#ifndef _LATTICE_MPI_H
#define _LATTICE_MPI_H
#include "./include.h"
namespace qcu
{
    template <typename T>
    int _MPI_Isend(const void *buf, int count, int dest,
                   int tag, MPI_Comm comm, MPI_Request *request);
    template <typename T>
    int _MPI_Irecv(void *buf, int count, int source,
                   int tag, MPI_Comm comm, MPI_Request *request);
    template <typename T>
    int _MPI_Sendrecv(const void *sendbuf, int sendcount,
                      int dest, int sendtag, void *recvbuf, int recvcount,
                      int source, int recvtag,
                      MPI_Comm comm, MPI_Status *status);
    template <typename T>
    int _MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                       MPI_Op op, MPI_Comm comm);
}
#endif