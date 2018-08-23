#include <iostream>
#include <vector>
#include <utility>
#include <mpi.h>

#include "Point.hh"
#include "Monomial.hh"
#include "FunctionRn.hh"
#include "GradientDescent.hh"
#include "Newton.hh"
#include "QuasiNewton.hh"
#include "Unconstrained_Min.hh"
#include "MPI_helpers.hh"
#include "Dense_Matrix.hh"


// Unconstrained minimization of the Beale Function
// N.B. Gradient method too slow!
int
main (int argc, char *argv[]){

  MPI_Init (&argc, &argv);


  // Construction of the function that has to be minimized
  Monomial x1 (14.20312, {0, 0});
  Monomial x2 (3, {2, 0});
  Monomial x3 (-12.75, {1, 0});
  Monomial x4 (3, {1, 1});
  Monomial x5 (-2, {2, 1});
  Monomial x6 (1, {2, 4});
  Monomial x7 (4.5, {1, 2});
  Monomial x8 (-1, {2, 2});
  Monomial x9 (1, {2, 6});
  Monomial x10 (5.25, {1, 3});
  Monomial x11 (-2, {2, 3});



  std::vector<Monomial> terms;
  terms.push_back (x1);
  terms.push_back (x2);
  terms.push_back (x3);
  terms.push_back (x4);
  terms.push_back (x5);
  terms.push_back (x6);
  terms.push_back (x7);
  terms.push_back (x8);
  terms.push_back (x9);
  terms.push_back (x10);
  terms.push_back (x11);



  FunctionRn f;
  for (const Monomial & m: terms)
      f.addMonomial (m);



  // Construction of the identity matrix as an approximation of the inverse of the Hessian for the QuasiNewton method

  la::Dense_Matrix Id (2,2,0);
  for (unsigned i = 0; i < 2; i++){
    Id (i,i) = 1;
  }

  // Initial point

  Point P0 ({1.2, .2});

  // Optimization
  Newton<FunctionRn> newton_method (f, 100, 1e-5); // For tolerance there is a default
  QuasiNewton<FunctionRn> quasiN_method (f, Id, 100, 1e-5);
  GradientDescent<FunctionRn> grad_method (f, 100, 1e-5);

  Unconstrained_Min<Newton<FunctionRn>> solver_Newton (newton_method);
  Unconstrained_Min<QuasiNewton<FunctionRn>> solver_quasiN (quasiN_method);
  Unconstrained_Min<GradientDescent<FunctionRn>> solver_grad (grad_method);

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
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------QuasiNewton--------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_quasiN.first << '\n';
    std::cout << "-iterations = " << x_quasiN.second << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Gradient-----------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_grad.first << '\n';
    std::cout << "-iterations = " << x_grad.second << '\n';
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
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------QuasiNewton--------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_quasiN_multistart.first << '\n';
    std::cout << "-iterations = " << x_quasiN_multistart.second << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Gradient-----------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_grad_multistart.first << '\n';
    std::cout << "-iterations = " << x_grad_multistart.second << '\n';
  }

  MPI_Finalize ();
  return 0;

}
