#include <mpi.h>

#include "MPI_helpers.hh"

namespace mpi
{
  int
  rank (void)
  {
    int rk;
    MPI_Comm_rank (MPI_COMM_WORLD, &rk);
    return rk;
  }

  int
  size (void)
  {
    int sz;
    MPI_Comm_size (MPI_COMM_WORLD, &sz);
    return sz;
  }
}
