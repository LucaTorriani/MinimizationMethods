#include <sstream>
#include <string>

#include "Dense_Matrix.hh"

namespace la
{
  Dense_Matrix::Dense_Matrix (size_type rows, size_type columns,
    const_reference value)
    : m_rows (rows), m_columns (columns),
    m_data (m_rows * m_columns, value) {}

    Dense_Matrix::Dense_Matrix (const Point & P ):
      m_rows (P.get_n_dimensions ()), m_columns (1), m_data (P.get_coords ()) {}

    Dense_Matrix::Dense_Matrix (std::istream & in)
    {
      read (in);
    }

  Dense_Matrix::size_type
  Dense_Matrix::sub2ind (size_type i, size_type j) const
  {
    return i * m_columns + j;
  }

  void
  Dense_Matrix::read (std::istream & in)
  {
    std::string line;
    std::getline (in, line);

    std::istringstream first_line (line);
    first_line >> m_rows >> m_columns;
    m_data.resize (m_rows * m_columns);

    for (size_type i = 0; i < m_rows; ++i)
      {
        std::getline (in, line);
        std::istringstream current_line (line);

        for (size_type j = 0; j < m_columns; ++j)
          {
            current_line >> (*this)(i, j);
          }
      }
  }

  void
  Dense_Matrix::swap (Dense_Matrix & rhs)
  {
    using std::swap;
    swap (m_rows, rhs.m_rows);
    swap (m_columns, rhs.m_columns);
    swap (m_data, rhs.m_data);
  }

  Dense_Matrix::reference
  Dense_Matrix::operator () (size_type i, size_type j)
  {
    return m_data[sub2ind (i, j)];
  }

  Dense_Matrix::const_reference
  Dense_Matrix::operator () (size_type i, size_type j) const
  {
    return m_data[sub2ind (i, j)];
  }

  Dense_Matrix::size_type
  Dense_Matrix::rows (void) const
  {
    return m_rows;
  }

  Dense_Matrix::size_type
  Dense_Matrix::columns (void) const
  {
    return m_columns;
  }

  Dense_Matrix
  Dense_Matrix::transposed (void) const
  {
    Dense_Matrix At (m_columns, m_rows);

    for (size_type i = 0; i < m_columns; ++i)
      for (size_type j = 0; j < m_rows; ++j)
        At (i, j) = operator () (j, i);

    return At;
  }


  Dense_Matrix
  Dense_Matrix::operator/= (double c)
  {
    for (size_type i = 0; i < m_rows; i++)
      for (size_type j = 0; j < m_rows; j++)
         (*this) (i, j) /= c;
    return *this;
  }

  std::ostream & operator<< (std::ostream & os, const Dense_Matrix & C)
  {
    using size_type = Dense_Matrix::size_type;
    for (size_type i = 0; i < C.m_rows; i++)
    {
      for (size_type j = 0; j < C.m_columns; j++)
      {
        os << C (i,j) << " ";
      }
      os << '\n';
    }

    return os;
  }


  Point operator* (const Dense_Matrix & A, const Point & p1)
  {
    using size_type = Dense_Matrix::size_type;
    Point p2 (A.m_rows);
    for (size_type i = 0; i < A.m_rows; ++i)
    {
      double temp(0.0);
      for (size_type j = 0; j < A.m_columns; ++j)
        temp +=  A (i,j) * p1.get_coord (j);
      p2.set_coord (i, temp );
    }
    return p2;
  }

  Point operator* (const Point & p1, const Dense_Matrix & A)
  {
    using size_type = Dense_Matrix::size_type;
    Point p2 (A.m_columns);
    for (size_type j = 0; j < A.m_columns; ++j)
    {
      double temp(0.0);
      for (size_type i = 0; i < A.m_rows; ++i)
        temp +=  A (i,j) * p1.get_coord (i);
      p2.set_coord (j, temp );
    }
    return p2;
  }

  Dense_Matrix operator* (double c, const Dense_Matrix & M)
  {
    using size_type = Dense_Matrix::size_type;
    Dense_Matrix C (M.m_rows, M.m_columns);
    for (size_type i = 0; i < M.m_rows; ++i)
      for (size_type j = 0; j < M.m_columns; ++j)
        C (i,j) = c * M (i,j);

  return C;
  }

  Dense_Matrix
  operator/ (const Dense_Matrix & A, double c)
  {
    using size_type = Dense_Matrix::size_type;
    Dense_Matrix B (A.m_rows, A.m_columns);
    for (size_type i = 0; i < A.m_rows; ++i)
      for (size_type j = 0; j < A.m_columns; ++j)
        B (i, j) = A (i, j) / c;

    return B;
  }

  Dense_Matrix
  operator+ (const Dense_Matrix & A, const Dense_Matrix & B)
  {
    using size_type = Dense_Matrix::size_type;
    Dense_Matrix C (A.m_rows, A.m_columns);
      for (size_type i = 0; i < A.m_rows; ++i)
        for (size_type j = 0; j < A.m_columns; ++j)
          C (i,j) = A (i,j) + B (i,j);

    return C;
  }

  Dense_Matrix
  operator- (const Dense_Matrix & A, const Dense_Matrix & B)
  {
    using size_type = Dense_Matrix::size_type;
    Dense_Matrix C (A.m_rows, A.m_columns);
    for (size_type i = 0; i < A.m_rows; ++i)
      for (size_type j = 0; j < A.m_columns; ++j)
        C(i,j) = A(i,j) - B(i,j);

    return C;
  }

  Dense_Matrix
  operator* (const Dense_Matrix & A, const Dense_Matrix & B)
  {
    using size_type = Dense_Matrix::size_type;

    Dense_Matrix C (A.m_rows, B.m_columns);

    for (size_type i = 0; i < A.m_rows; ++i)
    for (size_type j = 0; j < B.m_columns; ++j)
    for (size_type k = 0; k < A.m_columns; ++k)
    C (i, j) += A (i, k) * B (k, j);

    return C;
  }

  Dense_Matrix
  prod_tens (const Point & p1, const Point & p2)
  {
    using size_type = Dense_Matrix::size_type;
    Dense_Matrix A (p2.get_n_dimensions (), p1.get_n_dimensions ());
    for (size_type i = 0; i < A.m_rows; ++i)
      for (size_type j = 0; j < A.m_columns; ++j)
        A(i, j) = p1.get_coord (i) * p2.get_coord (j);

    return A;
  }

  void
  swap (Dense_Matrix & A, Dense_Matrix & B)
  {
    A.swap (B);
  }

}
