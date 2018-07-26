#include <iostream>
#include <vector>
#include <utility>
#include <mpi.h>

#include "Point.hh"
#include "Monomial.hh"
#include "FunctionRn.hh"
#include "FunctionRn_Constrained.hh"
#include "GradientDescent.hh"
#include "Newton.hh"
#include "QuasiNewton.hh"
#include "Constrained_Min.hh"
#include "UnConstrained_Min.hh"
#include "MPI_helpers.hh"
#include "Dense_Matrix.hh"

int
main (int argc, char *argv[]){

  MPI_Init (&argc, &argv);

  // EXAMPLE 7.18 pag 268 (Calcolo Scientico. Quarteroni, Saleri, Gerivasio, 5ed)
  // Minimization of f(x) subject to the constraint h=0
  // N.B QuasiNewton ok, Newton very slow, Gradient diverges

  // Construction of the function that has to be minimized
  Monomial x1 (0.6, {2, 0});
  Monomial x2 (0.5, {1, 1});
  Monomial x3 (-1, {0, 1});
  Monomial x4 (3, {1, 0});


  std::vector<Monomial> terms;
  terms.push_back (x1);
  terms.push_back (x2);
  terms.push_back (x3);
  terms.push_back (x4);


  FunctionRn f;
  for (const Monomial & m: terms)
      f.addMonomial (m);


  // Construction of the constraints
  FunctionRn g, h;
  Monomial h_mon1 (1,{2,0});
  Monomial h_mon2 (1,{0,2});
  Monomial h_mon3 (-1,{0,0});


  h.addMonomial (h_mon1);
  h.addMonomial (h_mon2);
  h.addMonomial (h_mon3);

  // Construction of the penalization function

  FunctionRn_Constrained penal;
  penal.set_f (f);
  penal.add_h_constraint (h);

  // Construction of the identity matrix as an approximation of the inverse of the Hessian for the QuasiNewton method

  la::Dense_Matrix Id (2,2,0);
  for (unsigned i = 0; i < 2; i++){
    Id (i,i) = 1;
  }

  // Initial point

  Point P0 ({1.2, .2});

  // Optimization

  Constrained_Min<Newton<FunctionRn_Constrained>> solver_Newton (penal, 100, 100, 1e-5);
  Constrained_Min<QuasiNewton<FunctionRn_Constrained>> solver_quasiN (penal, Id, 100, 100, 1e-5);
  Constrained_Min<GradientDescent<FunctionRn_Constrained>> solver_grad (penal, 100, 100, 1e-5);

  //----------------------SEQUENTIAL-----------------------------------------------------------------------

  std::pair<Point, unsigned> x_Newton = solver_Newton.minimize (P0);
  std::pair<Point, unsigned> x_quasiN = solver_quasiN.minimize (P0);
  std::pair<Point, unsigned> x_grad = solver_grad.minimize (P0);

  if(mpi::rank() == 0)
  {
    std::cout << "-------------SEQUENTIAL---------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Newton-------------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_Newton.first << '\n';
    std::cout << "-iterations = " << x_Newton.second << '\n';
    std::cout << "-h(x_min) = " << h.eval (x_Newton.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------QuasiNewton--------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_quasiN.first << '\n';
    std::cout << "-iterations = " << x_quasiN.second << '\n';
    std::cout << "-h(x_min) = " << h.eval (x_quasiN.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Gradient-----------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_grad.first << '\n';
    std::cout << "-iterations = " << x_grad.second << '\n';
    std::cout << "-h(x_min) = " << h.eval (x_grad.first) << std::endl;
    std::cout <<  '\n' << std::endl;
  }

  //----------------------PARALLEL-----------------------------------------------------------------------


  unsigned n_trials = 10;

  std::pair<Point, unsigned> x_Newton_multistart = solver_Newton.minimize_multistart (n_trials, {-2, -2}, {2, 2});
  std::pair<Point, unsigned> x_quasiN_multistart = solver_quasiN.minimize_multistart (n_trials, {-2, -2}, {2, 2});
  std::pair<Point, unsigned> x_grad_multistart = solver_grad.minimize_multistart (n_trials, {-2, -2}, {2, 2});

  if(mpi::rank () == 0)
  {
    std::cout << "-------------PARALLEL-----------------------------------------------------------" << '\n';
    std::cout << "n_trials = " << n_trials << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Newton-------------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_Newton_multistart.first << '\n';
    std::cout << "-iterations = " << x_Newton_multistart.second << '\n';
    std::cout << "-h(x_min) = " << h.eval(x_Newton_multistart.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------QuasiNewton--------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_quasiN_multistart.first << '\n';
    std::cout << "-iterations = " << x_quasiN_multistart.second << '\n';
    std::cout << "-h(x_min) = " << h.eval(x_quasiN_multistart.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Gradient-----------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_grad_multistart.first << '\n';
    std::cout << "-iterations = " << x_grad_multistart.second << '\n';
    std::cout << "-h(x_min) = " << h.eval(x_grad_multistart.first) << std::endl;
  }

  MPI_Finalize ();
  return 0;

}
