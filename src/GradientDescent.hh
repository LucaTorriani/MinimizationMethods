#ifndef _HH_GRADIENT_HH__
#define _HH_GRADIENT_HH__

#include <cmath>
#include <utility>

#include "Dense_Matrix.hh"
#include "Backtrack.hh"

template <class F>
class GradientDescent {

protected:
  F f; // F can be either FunctionRn or FunctionRn_Constrained
  unsigned max_iter;
  double tolerance;

  // Function useful in the constrained minimization algorithm 
  void set_tolerance (double new_tol);

public:
  // Constructor: func is the function to be minimized, max_it is the maximum
  // number of iterations imposed to the method and tol is the tolerance;
  GradientDescent (const F & func, unsigned max_it, double tol = 1e-5)
    : f (func), max_iter (max_it), tolerance (tol) {};

  // Minimizes the data member f starting from the initial point P;
  std::pair<Point, unsigned> solve (const Point & P) const;
};

template <class F>
void GradientDescent<F>::set_tolerance (double new_tol)
{
  tolerance = new_tol;
}

template<class F>
std::pair<Point, unsigned> GradientDescent<F>::solve (const Point & P) const
{
  Point new_x (P.get_coords ());
  unsigned it = 0;
  Point grad = f.compute_gradient (new_x);
  bool converged = false;

  while(it < max_iter && !converged)
  {
    it++;
    Point d  = -1 * grad;
    new_x = backtrack (f, new_x, grad, d);
    grad = f.compute_gradient (new_x);
    converged =  grad.infinity_norm () < tolerance;
  }

  return std::make_pair (new_x, it);
}

#endif
