#include "FunctionRn.hh"

// #include <iostream>

double FunctionRn::eval (const Point & P) const
{
  double value = 0;

  for (std::size_t i = 0; i < monoms.size (); ++i)
    value += monoms[i].eval (P);

  return value;
}


void FunctionRn::addMonomial (const Monomial & m)
{
  monoms.push_back (m);
}

//evaluate derivative wrt dim j in P
double FunctionRn::eval_deriv (std::size_t j, const Point & P) const
{
  // Second order finite differences
  Point P1 (P), P2 (P);
  P1.set_coord (j, P.get_coord (j) + h);
  P2.set_coord (j, P.get_coord (j) - h);
  return (eval (P1) - eval (P2)) / (2 * h);
}


Point FunctionRn::compute_gradient (const Point & P0) const
{
  std::vector<double> grad;
  for (std::size_t j = 0; j < P0.get_n_dimensions (); ++j)
    grad.push_back (eval_deriv (j, P0));

  return Point (grad);
}

double FunctionRn::eval_2deriv (std::size_t i, std::size_t j, const Point & P) const
{
  // Second order finite differences
  double result = 0.0;
  Point P1 (P), P2 (P), P3 (P), P4 (P), P5 (P), P6 (P);

  if (i != j)
  {
    P1.set_coord (i, P.get_coord (i) + h);
    P1.set_coord (j, P.get_coord (j) + k);

    P2.set_coord (i, P.get_coord (i) +  h);

    P3.set_coord (j, P.get_coord (j) + k);

    P4.set_coord (i, P.get_coord (i) -  h);

    P5.set_coord (j, P.get_coord (j) -  k);

    P6.set_coord (i, P.get_coord (i) -  h);
    P6.set_coord (j, P.get_coord (j) - k);
    result = (eval (P1) - eval (P2) - eval (P3) +  2 * eval (P) - eval (P4) - eval (P5) + eval (P6)) / (2*h*k);
  }

  else
  {
    P1.set_coord (j, P.get_coord (j) - 2 * h);
    P2.set_coord (j, P.get_coord (j) - h);
    P3.set_coord (j, P.get_coord (j) + h);
    P4.set_coord (j, P.get_coord (j) + 2 * h);
    result = (-0.25 * eval (P1) + 4 * eval (P2) - 7.5 * eval (P) + 4 * eval (P3) - 0.25 * eval (P4)) / (3 * ( h * h ));
  }
  return result;
}

la::Dense_Matrix FunctionRn::compute_hessian (const Point & P) const
{
  std::size_t n = P.get_n_dimensions ();
  la::Dense_Matrix H (n, n, 0);
  for (std::size_t i = 0; i < n - 1; i++)
  {
    for (std::size_t j = i + 1; j < n; j++)
    {
      H (i, j) = eval_2deriv (i, j, P);
    }
  }

  H = H + H.transposed ();

  for (std::size_t i = 0; i < n; i++)
  {
    H (i, i) = eval_2deriv (i, i, P);
  }

  return H;
}
