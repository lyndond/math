#ifndef STAN_MATH_PRIM_FUN_DIAG_PRE_MULTIPLY_TRI_HPP
#define STAN_MATH_PRIM_FUN_DIAG_PRE_MULTIPLY_TRI_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Return the product of the diagonal matrix formed from a vector
 * or row_vector and a lower-triangular matrix.
 *
 * @tparam T1 type of the vector/row_vector
 * @tparam T2 type of the lower-triangular matrix
 * @param m1 input vector/row_vector
 * @param m2 input matrix
 *
 * @return product of the diagonal matrix formed from the
 * vector or row_vector and a lower-triangular matrix.
 */
template <typename T1, typename T2, require_eigen_vector_t<T1>* = nullptr,
          require_eigen_t<T2>* = nullptr,
          require_all_not_st_var<T1, T2>* = nullptr>
auto diag_pre_multiply_tri(const T1& m1, const T2& m2) {
  check_size_match("diag_pre_multiply_tri", "m1.size()", m1.size(), "m2.rows()",
                   m2.rows());
  check_square("diag_pre_multiply_tri", "m2", m2);
  check_lower_triangular("diag_pre_multiply_tri", "m2", m2);

  const int n = m2.rows();
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(n, n);
  
  for (size_t r=0; r<m1.size(); ++r) {
    // segment
    res.row(r).segment(0, r+1) = m1(r) * m2.row(r).segment(0,r+1);
  }

  return res;
}

}  // namespace math
}  // namespace stan
#endif
