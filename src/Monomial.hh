#ifndef MONOMIAL_H_
#define MONOMIAL_H_

#include <vector>

#include "Point.hh"

class Monomial
{
  double coeff;
  std::vector<double> powers;

public:
  Monomial (double c, std::vector<double> const & pows)
    : coeff (c), powers (pows) {}
  double eval (Point const & P) const; // evaluate the monomial in P
};

#endif /* MONOMIAL_H_ */
