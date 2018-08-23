#ifndef UNCOSTRAINED_MIN_
#define UNCOSTRAINED_MIN_

#include <vector>
#include <random>
#include <utility>
#include <mpi.h>

#include "FunctionRn.hh"
#include "Dense_Matrix.hh"
#include "Point.hh"
#include "MPI_helpers.hh"
#include "Static_Constexpr.hh"


template <class Policy>
class Unconstrained_Min : public Policy
{

public:

  Unconstrained_Min (const Policy & p): Policy (p) {};

  // Minimizes the function fun
  // starting from the initial point P0 and calling the Policy method Policy::solve(P0).
  // It returns a pair with the point in which the function achieves its minimum
  // value and the number of iterations needed to reach convergence.
  std::pair<Point, unsigned>
  minimize (const Point & P0) const;

  // Minimize_multistart is thought to be used in parallel.
  // It minimizes the function fun
  // starting from a sequence of random points whose number is passed as
  // argument in n_trials. The sequence of random points is generated inside the square
  // [x1, x2] x [y1, y2]; the vector inf_limits passed as argument contains the
  // coordinates x1 and y1, while the vector sup_limits containes the coordinates x2 and y2.
  // The sequence of point is subdivided among the processes, and each process
  // calls the Policy method Policy::solve(P0) staring from each P0 assigned to it.
  // Each process saves its best minimum and in the end they compare their results, and the
  // proccess that has found the point in which the function reaches its minimum
  // sends the point and the number of iterations needed to compute it to all the
  // other processes. Every process returns a pair with the minimum point and
  // the number of iterations.
  std::pair<Point, unsigned>
  minimize_multistart (unsigned n_trials, const std::vector<double> & inf_limits,
                       const std::vector<double> & sup_limits) const;
};

template <class Policy>
std::pair<Point, unsigned> Unconstrained_Min<Policy>::minimize (const Point & P0) const
{
  return (Policy::solve (P0));
}

template <class Policy>
std::pair<Point, unsigned>
Unconstrained_Min<Policy>::minimize_multistart (unsigned n_trials,
                                                const std::vector<double> & inf_limits,
                                                const std::vector<double> & sup_limits) const
{
  const unsigned rank = mpi::rank ();
  std::default_random_engine generator (rank * big_number);
  std::uniform_real_distribution<double> distribution (0, 1);
  std::vector<double> random_coords;
  for (std::size_t i = 0; i < inf_limits.size (); ++i)
    {
      const double rand_val = distribution (generator);
      random_coords.push_back (inf_limits[i] +
                               (sup_limits[i] - inf_limits[i])
                               * rand_val);
    }

  Point random_point = Point (random_coords);

  std::pair<Point, unsigned> p_min = Policy::solve (random_point);
  unsigned iteration = p_min.second;
  double f_min = Policy::f.eval (p_min.first);

  const unsigned size = mpi::size ();
  const unsigned portion = n_trials / size;
  const unsigned my_trials = rank < n_trials % size ? portion + 1 : portion;
  for (unsigned n = 1; n < my_trials; ++n)
    {
      for (std::size_t i = 0; i < inf_limits.size (); ++i)
        {
          const double rand_val = distribution (generator);
          random_coords[i] = inf_limits[i] +
            (sup_limits[i] - inf_limits[i]) * rand_val;
        }

      random_point = Point (random_coords);
      const std::pair<Point, unsigned> p_new = Policy::solve (random_point);
      const double f_new = Policy::f.eval (p_new.first);
      if (f_new < f_min)
        {
          p_min.first = p_new.first;
          iteration = p_new.second;
          f_min = f_new;
        }
    }

  std::pair<double, int> minimum (f_min, rank);

  MPI_Allreduce (MPI_IN_PLACE, & minimum, 1, MPI_DOUBLE_INT,
                 MPI_MINLOC, MPI_COMM_WORLD);

  std::vector<double> min_coords = p_min.first.get_coords ();
  MPI_Bcast (min_coords.data (), min_coords.size (), MPI_DOUBLE,
             minimum.second, MPI_COMM_WORLD);

  MPI_Bcast (& iteration, 1, MPI_UNSIGNED, minimum.second, MPI_COMM_WORLD);

  return std::make_pair (Point (min_coords), iteration);
}

#endif
