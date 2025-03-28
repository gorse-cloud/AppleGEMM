#pragma onace

#include <random>
#include <vector>

std::vector<float> random_matrix(uint64_t m, uint64_t n);

void matmul(const float *a, const float *b, float *c, uint64_t m, uint64_t n,
            uint64_t k);
