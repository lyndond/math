#ifndef STAN_MATH_REV_FUN_DIAG_PRE_MULTIPLY_TRI_HPP
#define STAN_MATH_REV_FUN_DIAG_PRE_MULTIPLY_TRI_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/diag_pre_multiply_tri.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {
namespace math {

/**
 * Compute coefficient-wise product of matrix with lower-triangular matrix, and 
 * return its rowwise sum.
 * 
 * @tparam T1 type of matrix
 * @tparam T2 type of lower-triangular matrix
 * @param m input matrix
 * @param L input lower-triangular matrix
 * @return rowwise sum of the product of the matrix with a lower-triangular 
 * matrix
 */
template <typename T1, typename T2, require_matrix_t<T1>* = nullptr,
          require_matrix_t<T2>* = nullptr>//,
          // require_any_st_var<T1, T2>* = nullptr>
static auto cwise_product_rowwise_sum(const T1& m, const T2& L){

  const int n = L.rows();
  Eigen::VectorXd res(n);

  for(size_t r = 0; r < n; ++r)
    for(size_t c = 0; c <= r; ++c)
      res(r) += m(r, c) * L(r, c);

  return res;
}

/**
 * Return the product of the diagonal matrix formed from the vector
 * or row_vector and a matrix.
 *
 * @tparam T1 type of the vector/row_vector
 * @tparam T2 type of the matrix
 * @param m1 input vector/row_vector
 * @param m2 input matrix
 *
 * @return product of the diagonal matrix formed from the
 * vector or row_vector and a matrix.
 */
template <typename T1, typename T2, require_vector_t<T1>* = nullptr,
          require_matrix_t<T2>* = nullptr,
          require_any_st_var<T1, T2>* = nullptr>
auto diag_pre_multiply_tri(const T1& m1, const T2& L) {
  check_size_match("diag_pre_multiply_tri", "m1.size()", m1.size(), "L.rows()",
                   L.rows());
  check_square("diag_pre_multiply_tri", "L.rows() == L.cols()", value_of(L));
  check_lower_triangular("diag_pre_multiply_tri", "L", value_of(L));

  using inner_ret_type = decltype(diag_pre_multiply_tri(value_of(m1),
                                                        value_of(L)));
  using ret_type = return_var_matrix_t<inner_ret_type, T1, T2>;
  if (!is_constant<T1>::value && !is_constant<T2>::value) {
    arena_t<promote_scalar_t<var, T1>> arena_m1 = m1;
    arena_t<promote_scalar_t<var, T2>> arena_L = L;
    arena_t<ret_type> ret(diag_pre_multiply_tri(arena_m1.val(), 
                                                arena_L.val()));
    reverse_pass_callback([ret, arena_m1, arena_L]() mutable {
      arena_m1.adj() += cwise_product_rowwise_sum(ret.adj(), arena_L.val());
      arena_L.adj() += arena_m1.val().asDiagonal() * ret.adj();
    });
    return ret_type(ret);
  } else if (!is_constant<T1>::value) {
    arena_t<promote_scalar_t<var, T1>> arena_m1 = m1;
    arena_t<promote_scalar_t<double, T2>> arena_L = value_of(L);
    arena_t<ret_type> ret(diag_pre_multiply_tri(arena_m1.val(), 
                                                arena_L.val()));
    reverse_pass_callback([ret, arena_m1, arena_L]() mutable {
      arena_m1.adj() += cwise_product_rowwise_sum(ret.adj(), arena_L.val());
    });
    return ret_type(ret);
  } else if (!is_constant<T2>::value) {
    arena_t<promote_scalar_t<double, T1>> arena_m1 = value_of(m1);
    arena_t<promote_scalar_t<var, T2>> arena_L = L;
    arena_t<ret_type> ret(diag_pre_multiply_tri(arena_m1.val(),
                                                arena_L.val()));
    reverse_pass_callback([ret, arena_m1, arena_L]() mutable {
      arena_L.adj() += arena_m1.val().asDiagonal() * ret.adj();
    });
    return ret_type(ret);
  }
}

}  // namespace math
}  // namespace stan

#endif
