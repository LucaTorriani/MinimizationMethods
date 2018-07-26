#ifndef _HH_BACKTRACK_HH__
#define _HH_BACKTRACK_HH__

#include "Point.hh"

// Given the function to be minimized, the current solution xk, the gradient at xk
// and the direction dk it computes the new x
template <class F>
Point backtrack (const F & fun, const Point & xk, const Point & gk, const Point & dk, double rho = 0.25, double sigma = 1e-4)
{

  const double alphamin = 1.e-5;
  double alphak = 1.;
  double fk = fun.eval (xk);
  double prod_s_a = sigma * alphak;
  Point prod_gk_s_a = prod_s_a * gk;
  Point x = xk + alphak * dk;
  double scalar_prod = prod_gk_s_a * dk;

  while (fun.eval (x) > fk + scalar_prod && alphak > alphamin)
  {

    alphak = alphak * rho;
    Point prod_a_dk = alphak * dk;

    x = xk + prod_a_dk;

    prod_s_a = sigma * alphak;
    prod_gk_s_a = prod_s_a * gk;
    scalar_prod = prod_gk_s_a * dk;

  }

  return (x);

}

#endif
