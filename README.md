# AppleGEMM

[![CI](https://github.com/gorse-cloud/AppleGEMM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gorse-cloud/AppleGEMM/actions/workflows/ci.yml)

Hardware accelerated matrix multiplication implementations based on Apple Silicon AMX instructions.

```
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
MMBenchmark/Naive                     24081 ns        23830 ns        29393
MMBenchmark/Eigen                      8302 ns         8235 ns        84182
MMBenchmark/Armadillo                  2032 ns         2029 ns       349836
MMBenchmark/OpenBLAS                    692 ns          691 ns      1014890
MMBenchmark/AppleGEMM                   985 ns          982 ns       721679
MMBenchmarkTransposeA/Naive          167836 ns       167024 ns         4240
MMBenchmarkTransposeA/Eigen            8298 ns         8291 ns        84759
MMBenchmarkTransposeA/Armadillo        3831 ns         3789 ns       184639
MMBenchmarkTransposeA/OpenBLAS          454 ns          454 ns      1519354
MMBenchmarkTransposeA/AppleGEMM         467 ns          461 ns      1529376
MMBenchmarkTransposeB/Naive           79752 ns        79701 ns         8717
MMBenchmarkTransposeB/Eigen            8345 ns         8308 ns        84794
MMBenchmarkTransposeB/Armadillo        1597 ns         1588 ns       445165
MMBenchmarkTransposeB/OpenBLAS          878 ns          877 ns       784094
MMBenchmarkTransposeB/AppleGEMM        2178 ns         2177 ns       315223
MMBenchmarkTransposeAB/Naive         175866 ns       173931 ns         4044
MMBenchmarkTransposeAB/Eigen           8373 ns         8338 ns        84387
MMBenchmarkTransposeAB/Armadillo       2024 ns         2016 ns       346849
MMBenchmarkTransposeAB/OpenBLAS         723 ns          723 ns      1016482
MMBenchmarkTransposeAB/AppleGEMM       1703 ns         1693 ns       418325
```

Thanks [Peter Cawley's work on Apple's AMX](https://github.com/corsix/amx).
