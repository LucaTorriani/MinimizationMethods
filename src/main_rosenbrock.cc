#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <fstream>

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


// Rosenbrock function constrained with a cubic and a line
// f(x) = (1-x)^2 + 100*(y - x^2)^2
// s.t. (x-1)^3 - y + 1 <= 0 and x+y-2 <= 0
int main (int argc, char * argv[]){

  MPI_Init (& argc, & argv);

  // Construction of the function that has to be minimized

  Monomial x1 (1, {2, 0});
  Monomial x2 (-2, {1, 0});
  Monomial x3 (100, {0, 2});
  Monomial x4 (100, {4, 0});
  Monomial x5 (-200, {2, 1});
  Monomial x6 (1, {0, 0});

  std::vector<Monomial> terms;
  terms.push_back (x1);
  terms.push_back (x2);
  terms.push_back (x3);
  terms.push_back (x4);
  terms.push_back (x5);
  terms.push_back (x6);

  FunctionRn f;
  for (const Monomial & m: terms)
      f.addMonomial (m);

  // Construction of the contraints
  FunctionRn g1, g2;
  Monomial g1_mon1 (-1,{3,0});
  Monomial g1_mon2 (2,{0,0});
  Monomial g1_mon3 (3,{2,0});
  Monomial g1_mon4 (-3,{1,0});
  Monomial g1_mon5 (1,{0,1});

  g1.addMonomial (g1_mon1);
  g1.addMonomial (g1_mon2);
  g1.addMonomial (g1_mon3);
  g1.addMonomial (g1_mon4);
  g1.addMonomial (g1_mon5);

  Monomial g2_mon1 (-1,{1,0});
  Monomial g2_mon2 (-1,{0,1});
  Monomial g2_mon3 (2,{0,0});

  g2.addMonomial (g2_mon1);
  g2.addMonomial (g2_mon2);
  g2.addMonomial (g2_mon3);

  // Construction of the penalization function

  FunctionRn_Constrained penal;
  penal.set_f (f);
  penal.add_g_constraint (g1);
  penal.add_g_constraint (g2);

  // Construction of the identity matrix as an approximation of the inverse of the Hessian for the QuasiNewton method

  la::Dense_Matrix Id (2, 2, 0);
  for (unsigned i = 0; i < 2; i++){
    Id (i,i) = 1;
  }

  // Initial point

  Point P0 ({1.2, .2});

  // Optimization

  Constrained_Min<Newton<FunctionRn_Constrained>> solver_Newton (penal, 500, 100, 1e-5);
  Constrained_Min<QuasiNewton<FunctionRn_Constrained>> solver_quasiN (penal, Id, 500, 100, 1e-5);
  Constrained_Min<GradientDescent<FunctionRn_Constrained>> solver_grad (penal, 500, 100, 1e-5);

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
    std::cout << "-g1(x_min)  = " << g1.eval (x_Newton.first) << std::endl;
    std::cout << "-g2(x_min)  = " << g2.eval (x_Newton.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------QuasiNewton--------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_quasiN.first << '\n';
    std::cout << "-iterations = " << x_quasiN.second << '\n';
    std::cout << "-g1(x_min)  = " << g1.eval (x_quasiN.first) << std::endl;
    std::cout << "-g2(x_min)  = " << g2.eval (x_quasiN.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Gradient-----------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_grad.first << '\n';
    std::cout << "-iterations = " << x_grad.second << '\n';
    std::cout << "-g1(x_min)  = " << g1.eval (x_grad.first) << std::endl;
    std::cout << "-g2(x_min)  = " << g2.eval (x_grad.first) << std::endl;
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
    std::cout << "-g1(x_min)  = " << g1.eval (x_Newton_multistart.first) << std::endl;
    std::cout << "-g2(x_min)  = " << g2.eval (x_Newton_multistart.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------QuasiNewton--------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_quasiN_multistart.first << '\n';
    std::cout << "-iterations = " << x_quasiN_multistart.second << '\n';
    std::cout << "-g1(x_min)  = " << g1.eval (x_quasiN_multistart.first) << std::endl;
    std::cout << "-g2(x_min)  = " << g2.eval (x_quasiN_multistart.first) << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-------------Gradient-----------------------------------------------------------" << '\n';
    std::cout << "--------------------------------------------------------------------------------" << '\n';
    std::cout << "-x_min      = " << x_grad_multistart.first << '\n';
    std::cout << "-iterations = " << x_grad_multistart.second << '\n';
    std::cout << "-g1(x_min)  = " << g1.eval (x_grad_multistart.first) << std::endl;
    std::cout << "-g2(x_min)  = " << g2.eval (x_grad_multistart.first) << std::endl;
  }

  // Speed-Up Analysis
  //----------------------PARALLEL-----------------------------------------------------------------------

 //  unsigned j = 0;
 //  unsigned j_max = 10;
 //
 //  double mean_time_Newton = 0.;
 //  double mean_time_quasiNewton = 0.;
 //  double mean_time_grad = 0.;
 //
 //  std::ofstream myfile;
 //  myfile.open ("times.txt");
 //
 //  // Newton
 //  while (j < j_max)
 //  {
 //    auto tNewton_start = std::chrono::high_resolution_clock::now ();
 //    std::pair<Point, unsigned> x_Newton_multistart = solver_Newton.minimize_multistart (n_trials, {-1.5,-0.5}, {1.5, 2.5});
 //    auto tNewton_end = std::chrono::high_resolution_clock::now ();
 //    double t_Newton = std::chrono::duration<double> (tNewton_end-tNewton_start).count ();
 //    MPI_Allreduce (MPI_IN_PLACE, &t_Newton, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
 //    mean_time_Newton += t_Newton;
 //    if(mpi::rank () == 0){
 //      std::cout << "Required time for Newton execution: " << t_Newton << '\n';
 //      std::cout << "-x_min      = " << x_Newton_multistart.first << '\n';
 //      std::cout << "-iterations = " << x_Newton_multistart.second << '\n';
 //      myfile << t_Newton << std::endl;
 //    }
 //  j++;
 //  }
 //  if (mpi::rank () == 0)
 //  {
 //    mean_time_Newton /= j_max;
 //    std::cout << "Mean time: " << mean_time_Newton << std::endl;
 //    myfile << "\n"<< "Mean time: " << mean_time_Newton << std::endl;
 //  }
 //
 //  // QuasiNewton
 //  j = 0;
 //  while (j < j_max)
 //  {
 //  auto tquasiN_start = std::chrono::high_resolution_clock::now ();
 //  std::pair<Point, unsigned> x_quasiN_multistart = solver_quasiN.minimize_multistart (n_trials, {-1.5,-0.5}, {1.5, 2.5});
 //  auto tquasiN_end = std::chrono::high_resolution_clock::now ();
 //  double t_quasiN = std::chrono::duration<double> (tquasiN_end-tquasiN_start).count ();
 //  MPI_Allreduce (MPI_IN_PLACE, &t_quasiN, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
 //  mean_time_quasiNewton += t_quasiN;
 //   if(mpi::rank () == 0)
 //   {
 //     std::cout << "Required time for quasiNewton execution: " << t_quasiN << '\n';
 //     std::cout << "-x_min      = " << x_quasiN_multistart.first << '\n';
 //     std::cout << "-iterations = " << x_quasiN_multistart.second << '\n';
 //     myfile << t_quasiN << std::endl;
 //    }
 //  j++;
 //  }
 //  if (mpi::rank () == 0)
 //  {
 //    mean_time_quasiNewton /= j_max;
 //    std::cout << "Mean time: " << mean_time_quasiNewton << std::endl;
 //    myfile << "\n"<< "Mean time: " << mean_time_quasiNewton << std::endl;
 //  }
 //
 //
 //  // GradientDescent
 //  j = 0;
 //  while (j < j_max)
 // {
 //  auto tgrad_start = std::chrono::high_resolution_clock::now ();
 //  std::pair<Point, unsigned> x_grad_multistart = solver_grad.minimize_multistart (n_trials, {-1.5,-0.5}, {1.5, 2.5});
 //  auto tgrad_end = std::chrono::high_resolution_clock::now ();
 //  double t_grad = std::chrono::duration<double> (tgrad_end-tgrad_start).count ();
 //  MPI_Allreduce (MPI_IN_PLACE, &t_grad, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
 //  mean_time_grad += t_grad;
 //   if(mpi::rank () == 0)
 //   {
 //     std::cout << "Required time for GradientDescent execution: " << t_grad << '\n';
 //     std::cout << "-x_min      = " << x_grad_multistart.first << '\n';
 //     std::cout << "-iterations = " << x_grad_multistart.second << '\n';
 //     myfile << t_grad << std::endl;
 //   }
 //  j++;
 //  }
 //  if (mpi::rank () == 0)
 //  {
 //  mean_time_grad /= j_max;
 //  std::cout << "Mean time: " << mean_time_grad << std::endl;
 //  myfile << "\n"<< "Mean time: " << mean_time_grad << std::endl;
 //  }
 //
 //
 //  myfile.close ();

  MPI_Finalize ();
  return 0;

}
