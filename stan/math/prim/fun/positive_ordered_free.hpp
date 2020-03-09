#ifndef STAN_MATH_PRIM_FUN_POSITIVE_ORDERED_FREE_HPP
#define STAN_MATH_PRIM_FUN_POSITIVE_ORDERED_FREE_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Return the vector of unconstrained scalars that transform to
 * the specified positive ordered vector.
 *
 * <p>This function inverts the constraining operation defined in
 * <code>positive_ordered_constrain(Matrix)</code>,
 *
 * @tparam Vec type with a defined `operator[]`.
 * @param y Vector of positive, ordered scalars.
 * @return Free vector that transforms into the input vector.
 * @throw std::domain_error if y is not a vector of positive,
 *   ordered scalars.
 */
template <typename Vec, require_vector_like_t<Vec>* = nullptr>
inline auto positive_ordered_free(Vec&& y) {
  using std::log;
  check_positive_ordered("stan::math::positive_ordered_free",
                         "Positive ordered variable", y);
  auto k = y.size();
  plain_type_t<Vec> x(k);
  if (k == 0) {
    return x;
  }
  x[0] = log(y[0]);
  for (auto i = 1; i < k; ++i) {
    x[i] = log(y[i] - y[i - 1]);
  }
  return x;
}

}  // namespace math
}  // namespace stan

#endif
