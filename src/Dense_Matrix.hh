#ifndef Dense_Matrix_HH
#define Dense_Matrix_HH

#include <istream>
#include <vector>
#include <iostream>
#include <algorithm>

#include "Point.hh"

namespace la // Linear Algebra
{
  class Dense_Matrix final
  {
    typedef std::vector<double> container_type;

  public:
    typedef container_type::value_type value_type;
    typedef container_type::size_type size_type;
    typedef container_type::pointer pointer;
    typedef container_type::const_pointer const_pointer;
    typedef container_type::reference reference;
    typedef container_type::const_reference const_reference;

    friend std::ostream & operator<< (std::ostream& os, const Dense_Matrix & p1);
    friend Point operator* (const Dense_Matrix &, const Point &); // Product matrix*Point
    friend Point operator* (const Point &, const Dense_Matrix &); // Product Point*matrix
    friend Dense_Matrix operator* (double, const Dense_Matrix &); // Product constant*matrix
    friend Dense_Matrix operator/ (const Dense_Matrix & A, double c); // Matrix/constant
    friend Dense_Matrix operator+ (const Dense_Matrix &, const Dense_Matrix &);
    friend Dense_Matrix operator- (const Dense_Matrix &, const Dense_Matrix &);
    friend Dense_Matrix operator* (const Dense_Matrix &, const Dense_Matrix &);

    friend Dense_Matrix
    prod_tens (const Point & p1, const Point & p2); // columns*row

  private:
    size_type m_rows, m_columns;
    container_type m_data;

    size_type
    sub2ind (size_type i, size_type j) const; // return element (i,j)

  public:
    Dense_Matrix (void) = default;

    Dense_Matrix (size_type rows, size_type columns,
                  const_reference value = 0.0);

    explicit Dense_Matrix (const Point &);

    explicit Dense_Matrix (std::istream &);

    void
    read (std::istream &);

    void
    swap (Dense_Matrix &);

    reference
    operator () (size_type i, size_type j);
    const_reference
    operator () (size_type i, size_type j) const;

    size_type
    rows (void) const; // return the number of rows
    size_type
    columns (void) const; // return the number of columns

    Dense_Matrix
    transposed (void) const;

    Dense_Matrix
    operator/= (double);

  };

  std::ostream & operator<< (std::ostream& os, const Dense_Matrix & p);

  Point
  operator* (const Dense_Matrix &, const Point &);

  Point
  operator* (const Point &, const Dense_Matrix &);

  Dense_Matrix
  operator* (double, const Dense_Matrix &);

  Dense_Matrix
  operator/ (const Dense_Matrix & A, double c);

  Dense_Matrix
  operator+ (const Dense_Matrix &, const Dense_Matrix &);

  Dense_Matrix
  operator- (const Dense_Matrix &, const Dense_Matrix &);

  Dense_Matrix
  operator* (const Dense_Matrix &, const Dense_Matrix &);

  Dense_Matrix
  prod_tens (const Point & p1, const Point & p2);

  void
  swap (Dense_Matrix &, Dense_Matrix &); // Ricontrollare questo swap!!! Farlo o non farlo friend

  double
  mat_to_double (const Dense_Matrix &);
}

#endif // Dense_Matrix_HH
