#ifndef STAN_MATH_PRIM_SCAL_FUN_SUM_HPP
#define STAN_MATH_PRIM_SCAL_FUN_SUM_HPP

#include <stan/math/prim/meta.hpp>
namespace stan {
namespace math {

/**
 * Returns specified input value.
 *
 * @tparam T Type of element.
 * @param v Specified value.
 * @return Same value (the sum of one value).
 */
inline double sum(double v) { return v; }

/**
 * Returns specified input value.
 *
 * @tparam T Type of element.
 * @param v Specified value.
 * @return Same value (the sum of one value).
 */
inline int sum(int v) { return v; }

}  // namespace math
}  // namespace stan
#endif
#ifndef STAN_MATH_PRIM_ARR_FUN_SUM_HPP
#define STAN_MATH_PRIM_ARR_FUN_SUM_HPP

#include <stan/math/prim/meta.hpp>
#include <cstddef>
#include <vector>
#include <numeric>

namespace stan {
namespace math {

/**
 * Return the sum of the values in the specified standard vector.
 *
 * @tparam T Type of elements summed.
 * @param xs Standard vector to sum.
 * @return Sum of elements.
 */
template <typename T>
inline T sum(const std::vector<T>& xs) {
  return std::accumulate(xs.begin(), xs.end(), T{0});
}

}  // namespace math
}  // namespace stan
#endif
#ifndef STAN_MATH_PRIM_MAT_FUN_SUM_HPP
#define STAN_MATH_PRIM_MAT_FUN_SUM_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/arr/fun/sum.hpp>

namespace stan {
namespace math {

/**
 * Returns the sum of the coefficients of the specified
 * Eigen Matrix, Array or expression.
 *
 * @tparam Derived type of argument
 * @param v argument
 * @return Sum of coefficients of argument.
 */
template <typename Derived>
inline typename Eigen::DenseBase<Derived>::Scalar sum(
    const Eigen::DenseBase<Derived>& v) {
  return v.sum();
}

}  // namespace math
}  // namespace stan

#endif
