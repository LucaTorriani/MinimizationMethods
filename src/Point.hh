#ifndef POINT_H_
#define POINT_H_

#include <vector>
#include <fstream>

class Point
{
  typedef std::vector<double> coords_type;

protected:
  coords_type x;

public:
  friend Point operator+(const Point & p1, const Point & p2);
  friend Point operator*(const double c, const Point & p1); // constant*Point
  friend Point cw_prod(const Point & p1, const Point &p2); // Component-wise product
  friend Point operator-(const Point & p1, const Point & p2);
  friend double operator*(const Point & p1, const Point & p2); // Scalar product
  friend bool operator==(const Point & p1, const Point & p2);
  friend std::ostream& operator<<(std::ostream& os, const Point & p);
  friend bool check_dim(const Point & p1, const Point & p2);

  Point(void) = default;
  explicit Point (unsigned n) : x (n) {};
  explicit Point (coords_type const & coords): x (coords) {};

  //compute distance to Point p
  double distance (const Point & p) const;

  void print (void) const;

  std::size_t get_n_dimensions (void) const; // get the dimension of the point
  double get_coord (std::size_t i) const; // get the i-th coordinate
  void set_coord (std::size_t i, double val); // set the i-th coordinate equal to val
  coords_type get_coords (void) const; // get all the coordinates

  double euclidean_norm (void) const; // Euclidean norm
  double infinity_norm (void) const; // Infinity norm
};

Point operator+(const Point & p1, const Point & p2);
Point operator*(const double c, const Point & p1);
Point operator-(const Point & p1, const Point & p2);
double operator*(const Point & p1, const Point & p2);
Point cw_prod(const Point & p1, const Point & p2);
bool operator==(const Point & p1, const Point & p2);
std::ostream& operator<<(std::ostream& os, const Point & p);
bool check_dim(const Point & p1, const Point & p2);

#endif /* POINT_H_ */
