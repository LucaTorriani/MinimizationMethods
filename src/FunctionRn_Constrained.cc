#include "FunctionRn_Constrained.hh"

void FunctionRn_Constrained::set_f (const FunctionRn & f)
{
  func = f;
}


void FunctionRn_Constrained::add_g_constraint (const FunctionRn & g)
{
  g_constraints.push_back (g);
}


void FunctionRn_Constrained::add_h_constraint (const FunctionRn & h)
{
  h_constraints.push_back (h);
}

double FunctionRn_Constrained::eval (const Point & P) const
{
  double value = func.eval(P);
  for (const FunctionRn & g : g_constraints){
    value += (mu/2) * (std::max (-g.eval (P),0.0) * std::max (-g.eval (P), 0.0));
  }

  for (const FunctionRn & h : h_constraints)
  {
    value = value + mu/2*h.eval (P) * h.eval (P);

  }

  return value;
}

void FunctionRn_Constrained::set_mu (double m)
{
  mu = m;
}

double FunctionRn_Constrained::get_mu () const
{
  return (mu);
}

Point FunctionRn_Constrained::compute_gradient (const Point & P0) const
{
  std::vector<double> grad;

  for (std::size_t j = 0; j < P0.get_n_dimensions (); ++j)
    grad.push_back (eval_deriv (j, P0));

  return Point (grad);
}

double FunctionRn_Constrained::eval_deriv (std::size_t j, const Point & P) const
{
  double deriv_f = func.eval_deriv (j, P);
  double deriv_h = 0.0;
  double deriv_g = 0.0;

  for (const FunctionRn & h : h_constraints) {
    deriv_h += mu*h.eval (P) * h.eval_deriv (j, P);
  }

  for (const FunctionRn & g: g_constraints){
    if (g.eval (P) < 0)
    {
      deriv_g += mu * g.eval (P) * g.eval_deriv (j, P);
    }
  }

  return (deriv_f + deriv_g + deriv_h);
}



double FunctionRn_Constrained::eval_2deriv (std::size_t i, std::size_t j, const Point & P) const
{
  double deriv2_f = func.eval_2deriv (i,j,P);
  double deriv2_h = 0.0;
  double deriv2_g = 0.0;

  for(const FunctionRn & h: h_constraints){
    deriv2_h += mu * (h.eval_deriv (i, P)*h.eval_deriv(j,P) + h.eval(P)*h.eval_2deriv(i,j,P));
  }

  for(const FunctionRn & g: g_constraints){
    if(g.eval (P) < 0) {
        deriv2_g += mu * (g.eval_deriv (i, P) * g.eval_deriv (j,P) + g.eval (P) * g.eval_2deriv (i,j,P));
    }
  }
  
  return (deriv2_f + deriv2_g + deriv2_h);
}

la::Dense_Matrix FunctionRn_Constrained::compute_hessian (const Point & P) const
{
  std::size_t n = P.get_n_dimensions ();
  la::Dense_Matrix H(n, n, 0);
  for (std::size_t i = 1; i < n - 1 ; i++)
  {
    for (std::size_t j = i + 1; j < n; j++)
    {
      H(i,j) = eval_2deriv (i,j,P);
    }
  }

  H = H + H.transposed ();

  for (std::size_t i = 0; i < n; i++)
  {
    H(i,i) = eval_2deriv (i,i,P);
  }

  return H;
}
