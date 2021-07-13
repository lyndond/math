#ifndef STAN_MATH_REV_FUN_LOG1P_HPP
#define STAN_MATH_REV_FUN_LOG1P_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/fun/log1p.hpp>

namespace stan {
namespace math {

/**
 * The log (1 + x) function for variables (C99).
 *
 * The derivative is given by
 *
 * \f$\frac{d}{dx} \log (1 + x) = \frac{1}{1 + x}\f$.
 *
 * @tparam T Arithmetic or a type inheriting from `EigenBase`.
 * @param a The variable.
 * @return The log of 1 plus the variable.
 */
template <typename T>
inline auto log1p(const var_value<T>& a) {
  return make_callback_var(log1p(a.val()), [a](auto& vi) mutable {
    as_array_or_scalar(a.adj()) += as_array_or_scalar(vi.adj()) / as_array_or_scalar(1 + a.val());
  });
}

}  // namespace math
}  // namespace stan
#endif
