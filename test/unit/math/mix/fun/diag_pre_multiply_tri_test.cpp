#include <test/unit/math/test_ad.hpp>
#include <iostream>

void expect_diag_pre_multiply_tri(const Eigen::VectorXd& v,
                              const Eigen::MatrixXd& a) {
  auto f = [](const auto& x, const auto& y) {
    return stan::math::diag_pre_multiply_tri(x, y);
  };
  // stan::test::expect_ad(f, v, a);
  stan::test::expect_ad_matvar(f, v, a);
  // Eigen::RowVectorXd rv(v);
  // stan::test::expect_ad(f, rv, a);
  // stan::test::expect_ad_matvar(f, rv, a);
}

TEST(MathMixMatFun, diagPreMultiplyTri) {
  using stan::test::relative_tolerance;
  // 0 x 0
  // Eigen::MatrixXd a00(0, 0);
  // Eigen::VectorXd u0(0);
  // expect_diag_pre_multiply_tri(u0, a00);

  // 1 x 1
  Eigen::MatrixXd a11(1, 1);
  a11 << 10;
  Eigen::VectorXd u1(1);
  u1 << 3;
  expect_diag_pre_multiply_tri(u1, a11);

  // // 2 x 2
  Eigen::MatrixXd a22(2, 2);
  a22 << 1, 0, 
        100, 1000;
  Eigen::VectorXd u2(2);
  u2 << 2, 3;
  expect_diag_pre_multiply_tri(u2, a22);

  // // 3 x 3
  // Eigen::MatrixXd a33b(3, 3);
  // a33b << 1, 0, 0, 2, 3, 0, 4, 5, 6;
  // Eigen::VectorXd u3b(3);
  // u3b << 1, 2, 3;
  // expect_diag_pre_multiply_tri(u3b, a33b);

  // Eigen::MatrixXd a33c(3, 3);
  // a33c << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  // Eigen::VectorXd u3c(3);
  // u3c << 1, 2, 3;
  // expect_diag_pre_multiply_tri(u3c, a33c);

  // Eigen::MatrixXd a33(3, 3);
  // a33 << 1, 10, 100, 1000, 2, -4, 8, -16, 32;
  // Eigen::VectorXd u3(3);
  // u3 << -1.7, 111.2, -29.3;
  // expect_diag_pre_multiply_tri(u3, a33);

  // Eigen::MatrixXd a33d(3, 3);
  // a33d << 1, 0, 0, 0, 2, 0, 0, 0, 3;
  // Eigen::VectorXd u3d(3);
  // u3d << 1, 2, 3;
  // expect_diag_pre_multiply_tri(u3d, a33d);

  // // error: mismatched sizes
  // expect_diag_pre_multiply_tri(u2, a33);
  // expect_diag_pre_multiply_tri(u3, a22);

  // // non-square
  // Eigen::MatrixXd b23(2, 3);
  // b23 << 1, 2, 3, 4, 5, 6;
  // expect_diag_pre_multiply_tri(u2, b23);

  // Eigen::MatrixXd b32(3, 2);
  // b32 << 1, 2, 3, 4, 5, 6;
  // expect_diag_pre_multiply_tri(u3, b32);

  // Eigen::MatrixXd b13(1, 3);
  // b13 << 1, 2, 3;
  // expect_diag_pre_multiply_tri(u1, b13);

  // Eigen::MatrixXd b31(3, 1);
  // b31 << 1, 2, 3;
  // expect_diag_pre_multiply_tri(u3, b31);

  // // non-square error: mismatched sizes
  // expect_diag_pre_multiply_tri(u3d, b23);
}

// TEST(MathMixMatFun, diagPreMultiplyTriException) {
//   using Eigen::Dynamic;
//   using Eigen::Matrix;
//   using stan::math::diag_pre_multiply_tri;
//   Matrix<double, Dynamic, Dynamic> m(2, 2);
//   m << 2, 0,
//        4, 5;
//   Matrix<double, Dynamic, 1> v(3);
//   v << 1, 2, 3;
//   EXPECT_THROW(diag_pre_multiply_tri(v, m), std::invalid_argument);

//   Matrix<double, Dynamic, Dynamic> m_not_square(3, 4);
//   m_not_square << 1, 0, 0, 0, 
//                   5, 6, 0, 0, 
//                   9, 10, 11, 0;
//   EXPECT_THROW(diag_pre_multiply_tri(v, m_not_square), std::invalid_argument);

//   Matrix<double, Dynamic, Dynamic> m_not_lower_tri(2, 2);
//   m_not_lower_tri << 2, 3,
//                      4, 5;
//   Matrix<double, Dynamic, 1> v2(2);
//   v2 << 1, 2;
//   EXPECT_THROW(diag_pre_multiply_tri(v2, m_not_lower_tri), std::domain_error);
// }