# AppleGEMM

[![CI](https://github.com/gorse-cloud/AppleGEMM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gorse-cloud/AppleGEMM/actions/workflows/ci.yml)

Hardware accelerated matrix multiplication implementations based on Apple Silicon AMX instructions.

```
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
MMBenchmark/Naive                     24078 ns        23964 ns        29205
MMBenchmark/Eigen                      8258 ns         8247 ns        84888
MMBenchmark/Armadillo                  1971 ns         1970 ns       347783
MMBenchmark/OpenBLAS                    699 ns          694 ns      1032646
MMBenchmark/AppleGEMM                  1709 ns         1690 ns       413338
MMBenchmarkTransposeA/Naive          166716 ns       165310 ns         4157
MMBenchmarkTransposeA/Eigen            8400 ns         8340 ns        84226
MMBenchmarkTransposeA/Armadillo        3786 ns         3770 ns       187390
MMBenchmarkTransposeA/OpenBLAS          460 ns          456 ns      1539957
MMBenchmarkTransposeA/AppleGEMM        1536 ns         1536 ns       447785
MMBenchmarkTransposeB/Naive           79662 ns        79620 ns         8303
MMBenchmarkTransposeB/Eigen            8294 ns         8274 ns        84975
MMBenchmarkTransposeB/Armadillo        1580 ns         1579 ns       445517
MMBenchmarkTransposeB/OpenBLAS          856 ns          855 ns       833830
MMBenchmarkTransposeB/AppleGEMM        2209 ns         2193 ns       316817
MMBenchmarkTransposeAB/Naive         170956 ns       164832 ns         4283
MMBenchmarkTransposeAB/Eigen           8313 ns         8310 ns        78142
MMBenchmarkTransposeAB/Armadillo       2001 ns         1997 ns       354142
MMBenchmarkTransposeAB/OpenBLAS         681 ns          681 ns      1034386
MMBenchmarkTransposeAB/AppleGEMM       1679 ns         1678 ns       411247
```

Thanks [Peter Cawley's work on Apple's AMX](https://github.com/corsix/amx).
