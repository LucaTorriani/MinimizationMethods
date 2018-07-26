#include <iostream>
#include <cmath>

#include "Linear_System.hh"

namespace la {

  std::pair<la::Dense_Matrix, la::Dense_Matrix> lu (const la::Dense_Matrix & A)
  {
    std::size_t n = A.rows ();
    std::size_t m = A.columns ();
    la::Dense_Matrix L (n, n, 0);
    la::Dense_Matrix U (A);
    if (n != m) std::cerr << "Rectangular matrix" << std::endl;
    else
    {
      for (std::size_t i = 0; i < n; i++)
      {
        L(i,i) = 1.;
      }
      for (std::size_t k = 0; k < n - 1; k++)
      {
        for (std::size_t i = k + 1; i < n; i++)
        {
          double mik = U (i, k) / U (k, k);
            L (i, k) = mik;
          for (std::size_t j = k + 1; j < n; j++)
          {
            U (i, j) = U (i, j) - mik * U (k, j);
          }
          U (i, k) = 0.;
        }
      }
    }
    return std::make_pair (L, U);
  }

  Point bwsub (const la::Dense_Matrix & U, const Point & b)
  {
    std::size_t n = b.get_n_dimensions ();

    Point x (std::vector<double> (n, 0));
    double min = std::abs (U (0, 0));
    for (std::size_t i = 1; i < n; i++)
    {
         min = std::min (std::abs (U (i, i)), min);
            if (min < 1e-6) std::cerr << "Matrice singolare" << std::endl;
    }
    x.set_coord (n - 1, b.get_coord (n-1) / U (n - 1, n - 1));

    for (std::size_t i = 0; i < n - 1; i++)
    {
      double somma = 0;
      for(std::size_t j = n - 1 - i; j < n; j++)
      {
        somma = somma + U (n - 2 - i, j) * x.get_coord (j);
      }
      x.set_coord (n - 2 - i, (b.get_coord (n - 2 - i) - somma) / U (n - 2 - i, n - 2 - i));
    }

    return x;
  }

  Point fwsub (const la::Dense_Matrix & L, const Point & b)
  {
    std::size_t n = b.get_n_dimensions ();
    Point x (std::vector<double> (n, 0));

    double min = std::abs (L (0, 0));
    for (std::size_t i = 1; i < n; i++){
       min = std::min (std::abs (L (i, i)), min);
       if (min < 1e-6)
        std::cerr << "Matrice singolare" << std::endl;
    }

    x.set_coord (0, b.get_coord (0) / L (0, 0));
    for (std::size_t i = 1; i < n ; i++)
    {
      double somma = 0 ;
      for (std::size_t j = 0; j <= i - 1; j++)
      {
        somma = somma + L (i, j) * x.get_coord (j);

      }
      x.set_coord (i, (b.get_coord (i)- somma) / L(i, i));
   }
   return x;
  }

  Point solve_linear_system (const la::Dense_Matrix & A, const Point & b)
  {
    std::pair<la::Dense_Matrix, la::Dense_Matrix> LU = la::lu (A);
    la::Dense_Matrix L = LU.first;
    la::Dense_Matrix U = LU.second;
    Point y = la::fwsub (L, b);
    Point x = la::bwsub (U, y);
    return x;
  }

}
