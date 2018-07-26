#ifndef FUNCTIONRNCONSTRAINED_H_
#define FUNCTIONRNCONSTRAINED_H_

#include <vector>

#include "FunctionRn.hh"
#include "Point.hh"
#include "Dense_Matrix.hh"

class FunctionRn_Constrained{

  static constexpr double h = 1e-5;
  FunctionRn func; // Function to be minimized
  std::vector<FunctionRn> g_constraints; // "greater" constraints
  std::vector<FunctionRn> h_constraints; // "equality" constraints
  double mu = 1;

public:
  void set_f (const FunctionRn & f); // set f
  void add_g_constraint (const FunctionRn & g); // add g constraint
  void add_h_constraint (const FunctionRn & h); // add h constraint
  double get_mu (void) const;
  void set_mu (double m);

  //Evaluation of: f(P) + mu/2 * sum{i = 1..p} (g_i(P))^2 + mu/2* sum{j = 1..q} (max{0, -g_j(P)})^2
  double eval (const Point & P) const;

  // evaluate partial derivative w.r.t. dimension j in point P of the expression above
  double eval_deriv (std::size_t j, const Point & P) const;
  Point compute_gradient (const Point & P) const; // compute the gradient

  // evaluate partial derivative w.r.t. dimension j in point P of the expression above
  double eval_2deriv (std::size_t i, std::size_t j, const Point & P) const;
  la::Dense_Matrix compute_hessian (const Point & P) const; // compute hessian matrix

};

#endif
