#pragma once

#include "armadillo"
#include "Eigen/Dense"

#include <vector>

std::vector<float> random_matrix(uint64_t m, uint64_t n);

Eigen::MatrixXf random_matrix_eigen(uint64_t m, uint64_t n);

arma::mat random_matrix_arma(uint64_t m, uint64_t n);

void mm(const float *a, const float *b, float *c, uint64_t m, uint64_t n,
            uint64_t k, bool transA, bool transB);
