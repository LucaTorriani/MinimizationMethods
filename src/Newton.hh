#ifndef _HH_NEWTON_HH__
#define _HH_NEWTON_HH__

#include <cmath>
#include <utility>

#include "Dense_Matrix.hh"
#include "Linear_System.hh"
#include "Backtrack.hh"

template <class F>
class Newton {
protected:
  F f; // F can be either FunctionRn or FunctionRn_Constrained
  unsigned max_iter;
  double tolerance;

  // Function useful in the constrained minimization algorithm
  void set_tolerance (double new_tol);

public:

  // Constructor: func is the function to be minimized, max_it is the maximum
  // number of iterations imposed to the method and tol is the tolerance;
  Newton(const F & func, unsigned max_it, double tol = 1e-5)
    : f (func), max_iter (max_it), tolerance (tol) {};

  // Minimizes the data member f starting from the initial point P;
  std::pair<Point, unsigned> solve (const Point & P) const;

};



template <class F>
void Newton<F>::set_tolerance (double new_tol)
{
  tolerance = new_tol;
}

template<class F>
std::pair<Point, unsigned> Newton<F>::solve (const Point & P) const
{
  Point new_x (P.get_coords ());
  bool converged = false;
  unsigned it = 0;
  Point new_grad = f.compute_gradient (new_x);

  while(it < max_iter && !converged)
  {
    it++;
    la::Dense_Matrix H = f.compute_hessian (new_x) ;
    Point d  = la::solve_linear_system (H, -1 * new_grad);

    new_x = backtrack (f, new_x, new_grad, d);
    new_grad = f.compute_gradient (new_x);

    Point xkt (new_x.get_n_dimensions ());
    for (unsigned i = 0; i < xkt.get_n_dimensions (); i++)
    {
      xkt.set_coord (i, std::max ( std::abs (new_x.get_coord (i)), 1.0));
    }
    converged = ((1.0 / std::max (std::abs (f.eval (new_x)), 1.0)) * cw_prod (new_grad,xkt)).infinity_norm () < tolerance;
  }

  return (std::make_pair (new_x, it));
}

#endif
