#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(MathMatrixPrimMat, diagPreMultiplyTri) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::diag_pre_multiply_tri;
  using stan::math::diag_pre_multiply;

  Matrix<double, Dynamic, Dynamic> m(1, 1);
  m << 3;

  Matrix<double, Dynamic, 1> v(1);
  v << 9;

  Matrix<double, Dynamic, Dynamic> v_m(1, 1);
  v_m << 9;

  EXPECT_MATRIX_FLOAT_EQ(v_m * m, diag_pre_multiply_tri(v, m));
  EXPECT_MATRIX_FLOAT_EQ(diag_pre_multiply(v, m), diag_pre_multiply_tri(v, m));
}

TEST(MathMatrixPrimMat, diagPreMultiplyTri2) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::diag_pre_multiply_tri;
  Matrix<double, Dynamic, Dynamic> m(3, 3);
  m << 1, 0, 0,
       4, 5, 0,
       7, 8, 9;

  Matrix<double, Dynamic, 1> v(3);
  v << 1, 2, 3;

  Matrix<double, Dynamic, Dynamic> v_m(3, 3);
  v_m << 1, 0, 0, 
         0, 2, 0, 
         0, 0, 3;

  EXPECT_MATRIX_FLOAT_EQ(v_m * m, diag_pre_multiply_tri(v, m));

  Matrix<double, 1, Dynamic> rv(3);
  rv << 1, 2, 3;
  EXPECT_MATRIX_FLOAT_EQ(v_m * m, diag_pre_multiply_tri(rv, m));
}

TEST(MathMatrixPrimMat, diagPreMultiplyTriException) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::diag_pre_multiply_tri;
  Matrix<double, Dynamic, Dynamic> m(2, 2);
  m << 2, 0,
       4, 5;
  Matrix<double, Dynamic, 1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(diag_pre_multiply_tri(v, m), std::invalid_argument);

  Matrix<double, Dynamic, Dynamic> m_not_square(3, 4);
  m_not_square << 1, 0, 0, 0, 
                  5, 6, 0, 0, 
                  9, 10, 11, 0;
  EXPECT_THROW(diag_pre_multiply_tri(v, m_not_square), std::invalid_argument);

  Matrix<double, Dynamic, Dynamic> m_not_lower_tri(2, 2);
  m_not_lower_tri << 2, 3,
                     4, 5;
  Matrix<double, Dynamic, 1> v2(2);
  v2 << 1, 2;
  EXPECT_THROW(diag_pre_multiply_tri(v2, m_not_lower_tri), std::domain_error);
}

TEST(MathMatrixPrimMat, diagPreMultiplyTriZero) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::diag_pre_multiply_tri;
  Eigen::VectorXd in1(0);
  Eigen::MatrixXd in2(0, 0);
  Eigen::MatrixXd in3;
  Eigen::MatrixXd in4(0, 0);
  in3 = diag_pre_multiply_tri(in1, in2);
  EXPECT_EQ(in4, in3);
}
