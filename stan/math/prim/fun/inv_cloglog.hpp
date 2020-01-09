#ifndef STAN_MATH_PRIM_SCAL_FUN_INV_CLOGLOG_HPP
#define STAN_MATH_PRIM_SCAL_FUN_INV_CLOGLOG_HPP

#include <stan/math/prim/meta.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * The inverse complementary log-log function.
 *
 * The function is defined by
 *
 * <code>inv_cloglog(x) = 1 - exp(-exp(x))</code>.
 *
 * This function can be used to implement the inverse link
 * function for complementary-log-log regression.
 *
 *
   \f[
   \mbox{inv\_cloglog}(y) =
   \begin{cases}
     \mbox{cloglog}^{-1}(y) & \mbox{if } -\infty\leq y \leq \infty \\[6pt]
     \textrm{NaN} & \mbox{if } y = \textrm{NaN}
   \end{cases}
   \f]

   \f[
   \frac{\partial\, \mbox{inv\_cloglog}(y)}{\partial y} =
   \begin{cases}
     \frac{\partial\, \mbox{cloglog}^{-1}(y)}{\partial y} & \mbox{if }
 -\infty\leq y\leq \infty \\[6pt] \textrm{NaN} & \mbox{if } y = \textrm{NaN}
   \end{cases}
   \f]

   \f[
   \mbox{cloglog}^{-1}(y) = 1 - \exp \left( - \exp(y) \right)
   \f]

   \f[
   \frac{\partial \, \mbox{cloglog}^{-1}(y)}{\partial y} = \exp(y-\exp(y))
   \f]
 *
 * @param x Argument.
 * @return Inverse complementary log-log of the argument.
 */
inline double inv_cloglog(double x) {
  using std::exp;
  return 1 - exp(-exp(x));
}

}  // namespace math
}  // namespace stan

#endif
#ifndef STAN_MATH_PRIM_MAT_FUN_INV_CLOGLOG_HPP
#define STAN_MATH_PRIM_MAT_FUN_INV_CLOGLOG_HPP

#include <stan/math/prim/mat/fun/exp.hpp>
#include <stan/math/prim/scal/fun/inv_cloglog.hpp>
#include <stan/math/prim/vectorize/apply_scalar_unary.hpp>

namespace stan {
namespace math {

/**
 * Structure to wrap inv_cloglog() so that it can be vectorized.
 *
 * @tparam T type of variable
 * @param x variable
 * @return 1 - exp(-exp(x)).
 */
struct inv_cloglog_fun {
  template <typename T>
  static inline T fun(const T& x) {
    return inv_cloglog(x);
  }
};

/**
 * Vectorized version of inv_cloglog().
 *
 * @tparam T type of container
 * @param x container
 * @return 1 - exp(-exp()) applied to each value in x.
 */
template <typename T, typename = require_not_eigen_vt<std::is_arithmetic, T>>
inline auto inv_cloglog(const T& x) {
  return apply_scalar_unary<inv_cloglog_fun, T>::apply(x);
}

/**
 * Version of inv_cloglog() that accepts Eigen Matrix or matrix expressions.
 * @tparam Derived derived type of x
 * @param x Matrix or matrix expression
 * @return 1 - exp(-exp()) applied to each value in x.
 */
template <typename Derived,
          typename = require_eigen_vt<std::is_arithmetic, Derived>>
inline auto inv_cloglog(const Eigen::MatrixBase<Derived>& x) {
  return (1 - exp(-exp(x.derived().array()))).matrix();
}

/**
 * Version of inv_cloglog() that accepts Eigen Array or array expressions.
 * @tparam Derived derived type of x
 * @param x Matrix or matrix expression
 * @return 1 - exp(-exp()) applied to each value in x.
 */
template <typename Derived,
          typename = require_eigen_vt<std::is_arithmetic, Derived>>
inline auto inv_cloglog(const Eigen::ArrayBase<Derived>& x) {
  return 1 - exp(-exp(x.derived()));
}

}  // namespace math
}  // namespace stan

#endif
