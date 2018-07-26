#ifndef FUNCTIONRN_H_
#define FUNCTIONRN_H_

#include <vector>

#include "Monomial.hh"
#include "Point.hh"
#include "Dense_Matrix.hh"

class FunctionRn
{
protected:

  std::vector<Monomial> monoms;

  static constexpr double k = 1e-5;
  static constexpr double h = 1e-5;

public:
  double eval (const Point & P) const; // evluate the function in P

  void addMonomial (const Monomial & m);

  // evaluate partial derivative w.r.t. dimension j in point P
  double eval_deriv (std::size_t j, const Point & P) const;

  // compute the gradient
  Point compute_gradient (const Point & P) const;

  // evaluates mixed second partial derivative w.r.t. dimenson i and j in point P
  double eval_2deriv (std::size_t i, std::size_t j, const Point & P) const;

  // compute Hessian matrix
  la::Dense_Matrix compute_hessian (const Point & P) const;

};

#endif /* FUNCTIONRN_H_ */
