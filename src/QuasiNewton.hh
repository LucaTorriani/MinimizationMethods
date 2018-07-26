#ifndef _HH_QUASINEWTON_HH__
#define _HH_QUASINEWTON_HH__

#include <cmath>
#include <utility>

#include "Dense_Matrix.hh"
#include "Backtrack.hh"

template <class F>
class QuasiNewton {
protected:
  F f; // F can be either FunctionRn or FunctionRn_Constrained
  la::Dense_Matrix invH0; // Approximation of the inverse of the Hessian matrix
  unsigned max_iter;
  double tolerance;

  // Update the approximation of the inverse of the Hessian of f, passed as
  // argument in inv_h_k, using the BFGS formula.
  la::Dense_Matrix update_invH (const la::Dense_Matrix & inv_H_k,
                                const Point & delta_k,
                                const Point & gamma_k) const;

 // Function useful in the constrained minimization algorithm
  void set_tolerance (double new_tol);

public:
  // Constructor: func is the function to be minimized, H0_inv is the
  // approximation of the inverse of the Hession of func, max_it is the maximum
  // number of iterations imposed to the method and tol is the tolerance;
  QuasiNewton (const F & func, const la::Dense_Matrix & H0_inv,
               unsigned max_it, double tol = 1e-5)
    : f (func), invH0 (H0_inv), max_iter (max_it), tolerance (tol) {};

  // Minimizes the data member f starting from the initial point P;
  std::pair<Point, unsigned> solve (const Point & P) const;

};



template <class F>
void QuasiNewton<F>::set_tolerance (double new_tol){
  tolerance = new_tol;
}

template <class F>
la::Dense_Matrix QuasiNewton<F>::update_invH (const la::Dense_Matrix & inv_H_k,
  const Point & delta_k, const Point & gamma_k) const
  {
  double d_g = delta_k * gamma_k;
  double g_H_g =  (gamma_k * (inv_H_k * gamma_k) ) / d_g;
  la::Dense_Matrix mat1 = la::prod_tens (delta_k, delta_k);
  la::Dense_Matrix mat2 = la::prod_tens ( (inv_H_k * gamma_k), delta_k);
  la::Dense_Matrix mat3 = la::prod_tens (delta_k, (gamma_k * inv_H_k));

  la::Dense_Matrix result  = ((1.0 + g_H_g) * mat1 - mat2 - mat3) / d_g;
  result = inv_H_k + result;

  return result;
}

template<class F>
std::pair<Point, unsigned> QuasiNewton<F>::solve (const Point & P) const
{
  Point x (P.get_coords ());
  Point new_x = x;
  la::Dense_Matrix invH = invH0;
  bool converged = false;
  unsigned it = 0;
  Point new_grad = f.compute_gradient (new_x);
  Point grad = new_grad;

  while(it < max_iter && !converged)
  {
    it++;
    Point d  = -1 * (invH * new_grad);

    new_x = backtrack (f, new_x, grad, d);
    new_grad = f.compute_gradient (new_x);

    Point s = new_x - x; // delta
    Point y = new_grad - grad; // gamma

    invH = update_invH (invH, s, y);

    grad = new_grad;
    x = new_x;

    Point xkt (new_x);
    for (unsigned i = 0; i < xkt.get_n_dimensions (); i++)
    {
      xkt.set_coord (i, std::max ( std::abs (new_x.get_coord (i)),1.0));
    }
    converged =  ((1.0/std::max (std::abs (f.eval (new_x)),1.0))*cw_prod (new_grad,xkt)).infinity_norm () < tolerance;
  }

  return (std::make_pair (new_x, it));
}

#endif
