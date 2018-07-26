#include <cmath>

#include "Monomial.hh"

double Monomial::eval (Point const & P) const
{
  double value = 1;

  for (std::size_t dim = 0; dim < powers.size (); ++dim)
    value *= pow (P.get_coord (dim), powers[dim]);

  return coeff * value;
}
