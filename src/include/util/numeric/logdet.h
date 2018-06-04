#pragma once

#include "Eigen/Cholesky"

template <typename MatrixType>
inline typename MatrixType::Scalar logdet(const MatrixType& M) {
  using namespace Eigen;
  using std::log;
  typedef typename MatrixType::Scalar Scalar;
  Scalar ld = 0;
  LLT<Matrix<Scalar,Dynamic,Dynamic>> chol(M);
  auto& U = chol.matrixL();
  for (unsigned i = 0; i < M.rows(); ++i) {
    ld += log(U(i, i));
  }
  ld *= 2;
  return ld;
}
