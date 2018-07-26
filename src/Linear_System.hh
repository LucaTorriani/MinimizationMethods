#ifndef LINEAR_SYSTEM_HH
#define LINEAR_SYSTEM_HH

#include "Dense_Matrix.hh"
#include "Point.hh"

namespace la
{

  std::pair<la::Dense_Matrix, la::Dense_Matrix> lu (const la::Dense_Matrix & A); // Compute the LU factorization
  Point bwsub (const la::Dense_Matrix & U, const Point & b); // Backward substitution
  Point fwsub (const la::Dense_Matrix & L, const Point & b); // Forward substitution
  Point solve_linear_system (const la::Dense_Matrix & A, const Point & b); // Solve Ax = b

}

#endif
