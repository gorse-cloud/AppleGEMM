#include "benchmark/benchmark.h"

static void BM_SomeFunction(benchmark::State& state) {

}

// Register the function as a benchmark
BENCHMARK(BM_SomeFunction);

BENCHMARK_MAIN();
