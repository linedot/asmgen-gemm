# asmgen-gemm: Generator for GEMM microkernels using asmgen

Code previously part of the uarch\_bench project

## Requirements:
- [asmgen](https://github.com/linedot/asmgen)
- c++ compiler
- BLAS library (to test against)
- fmtlib (output in tests)


## Run performance test  (example on AVX2-capable machine)
```
$ cd asmgen-gemm
$ python3 ./gemmerator.py --mr 8 --nr 6 -V 16 -M l1 -t double -T fma256 --bvec-strat dist1_boff --avec-strat postload --output-filename gemmbench_8_6.cpp gemmbench.cpp.in
$ clang++ -std=c++20 -Ofast -g -fopenmp -march=native -mtune=native -o gemmbench_8_6 gemmbench_8_6.cpp performance_counters_perf.cpp
$ ./gemmbench_8_6
Will perform 5979 iterations, reducing measurement overhead to 0.1%
Event: CYCLES - min: 286640; avg: 287538; max: 289944
Avg. 15.9697 FLOPS/cycle
```
With a sufficiently large microkernel you should see a number very close to 100% of the theoretical peak performance of your CPU

## Run correctness test (example on AVX2-capable machine, using BLIS as BLAS library)

```bash
$ cd asmgen-gemm/tests
$ python3 ../gemmerator.py --mr 8 --nr 6 -V 16 -U2 -M contiguous -t double -T fma256 --bvec-strat dist1_boff --avec-strat postload --output-filename naive_asmgen_gemm.cpp naive_asmgen_gemm.cpp.in
$ g++ -std=c++20 -Ofast -g -fopenmp -march=native -mtune=native -I/usr/include/blis/ naivegemm.cpp naive_asmgen_gemm.cpp -o naivegemm -lfmt -lblis-mt
$ ./naivegemm 
BLAS library finished in 95818us. (721.383 GFLOP/s)
Naive implementation finished in 1312114us. (52.679 GFLOP/s)
Naive asmgen implementation finished in 311049us. (222.220 GFLOP/s)
Testing correctness with epsilon=2.220446049250313e-13
Biggest difference: 4.88426e-15
OK:   Difference between BLAS library and naive-implementation within tolerance.
Biggest difference: 1.547e-14
OK:   Difference between BLAS library and naive-asmgen implementation within tolerance.
Biggest difference: 1.63961e-14
OK:   Difference between naive and naive-asmgen implementation within tolerance.
```

Output of correctness test is not suitable for performance-testing as the test fixture for asmgen ukernels is not optimized
