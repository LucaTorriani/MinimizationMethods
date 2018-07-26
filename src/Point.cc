#include <cmath>
#include <iostream>

#include "Point.hh"

Point operator+(const Point & p1, const Point & p2){
  if(!check_dim(p1, p2))
  {
    std::cerr << "Error operator+: points don't have the same dimension" << '\n';
  }
  unsigned int n = p1.get_n_dimensions();
  Point::coords_type coords(n);
  for(unsigned int i = 0; i < n; i++)
    coords[i] = p1.x[i] + p2.x[i];
  return Point(coords);
}

Point operator*(const double c, const Point & p1){
  unsigned int n = p1.get_n_dimensions();
  Point::coords_type coords(n);
  for(unsigned int i = 0; i < n; i++){
    coords[i] = c * p1.x[i];
  }
  return Point(coords);
}

Point operator-(const Point & p1, const Point & p2){
  if(!check_dim(p1, p2))
  {
    std::cerr << "Error operator-: points don't have the same dimension" << '\n';
  }
  unsigned int n = p1.get_n_dimensions();
  Point::coords_type coords(n);
  for(unsigned int i = 0; i < n; i++)
    coords[i] = p1.x[i] - p2.x[i];
  return Point(coords);
}

double operator*(const Point & p1, const Point & p2){
  if(!check_dim(p1, p2))
  {
    std::cerr << "Error operator*: points don't have the same dimension" << '\n';
  }
  double result = 0.0;
  for(unsigned int i = 0; i < p1.get_n_dimensions(); i++){
    result += p1.x[i]*p2.x[i];
  }
  return result;
}

Point cw_prod(const Point & p1, const Point & p2){
  if(!check_dim(p1, p2))
  {
    std::cerr << "Error operator*: points don't have the same dimension" << '\n';
  }
  unsigned int n = p1.get_n_dimensions();
  Point::coords_type coords(n);
  for(unsigned int i = 0; i < n; i++){
    coords[i] = p1.x[i] * p2.x[i];
  }
  return Point(coords);
}

bool operator==(const Point & p1, const Point & p2){
  if(check_dim(p1, p2))
    return p1.x == p2.x;
  else
    return 0;
}

std::ostream& operator<<(std::ostream& os, const Point & p){
  if(p.get_n_dimensions() == 1){
    return os << p.x[0];
  }
  os << "(";
  unsigned i = 0;
  for(i = 0; i < p.get_n_dimensions() - 1; i++){
    os << p.x[i] << ", ";
  }
  os << p.x[i] << ")";
  return os;
}

bool check_dim(const Point & p1, const Point & p2){
  return p1.get_n_dimensions() == p2.get_n_dimensions();
}

double Point::distance (const Point & p) const
{
  double dist = 0.0;

  for (std::size_t i = 0; i < x.size (); ++i)
    {
      const double delta = x[i] - p.x[i];
      dist += delta * delta;
    }

  return sqrt (dist);
}

void Point::print (void) const
{
  for (auto it = x.begin (); it != x.end (); ++it)
    {
      std::cout << *it;
      std::cout << " ";
    }

  std::cout << std::endl;
}

double Point::get_coord (std::size_t i) const
{
  return x[i];
}


void Point::set_coord (std::size_t i, double val)
{
  x[i] = val;
}

Point::coords_type Point::get_coords (void) const
{
  return x;
}

double Point::euclidean_norm (void) const
{
  static const std::vector<double> zero_vector (x.size (), 0.);
  static const Point origin (zero_vector);
  return distance (origin);
}

double Point::infinity_norm (void) const
{
  double max_value = std::abs (x[0]);

  for (std::size_t i = 1; i < x.size (); ++i)
    {
      const double next = std::abs (x[i]);
      if (next > max_value) max_value = next;
    }

  return max_value;
}

std::size_t Point::get_n_dimensions (void) const
{
  return x.size ();
}
